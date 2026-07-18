#include <arpa/inet.h>
#include <cuda_runtime.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <netdb.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <regex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

static volatile sig_atomic_t g_stop = 0;
static constexpr uint32_t MAX_CANDIDATES = 512;

static void on_sigint(int) {
    g_stop = 1;
}

#define CUDA_CHECK(x) do { \
    cudaError_t err__ = (x); \
    if (err__ != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err__)); \
        exit(1); \
    } \
} while (0)

struct Options {
    std::string pool = "stratum+tcp://127.0.0.1:3333";
    std::string address;
    std::string worker = "cuda";
    int device = -1;
    int blocks = 4096;
    int threads = 128;
    uint64_t batch = 524288;
    uint64_t seed = 0;
    int lane_index = 0;
    int lane_count = 1;
    bool dashboard = true;
    bool quiet = false;
    bool child = false;
    std::string stats_dir;
};

struct Candidate {
    uint32_t count;
    uint32_t cap;
    uint64_t nonce[MAX_CANDIDATES];
    uint8_t hash[MAX_CANDIDATES][32];
};

__device__ __forceinline__ uint32_t rotr32(uint32_t x, uint32_t n) {
    return (x >> n) | (x << (32u - n));
}

__device__ __forceinline__ void gmix(
    uint32_t &a, uint32_t &b, uint32_t &c, uint32_t &d, uint32_t mx, uint32_t my) {
    a = a + b + mx; d = rotr32(d ^ a, 16);
    c = c + d;      b = rotr32(b ^ c, 12);
    a = a + b + my; d = rotr32(d ^ a, 8);
    c = c + d;      b = rotr32(b ^ c, 7);
}

__device__ __forceinline__ uint32_t load32(const uint8_t *p) {
    return ((uint32_t)p[0]) | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

__device__ __forceinline__ void store32(uint8_t *p, uint32_t x) {
    p[0] = (uint8_t)x;
    p[1] = (uint8_t)(x >> 8);
    p[2] = (uint8_t)(x >> 16);
    p[3] = (uint8_t)(x >> 24);
}

__device__ void blake3_oneblock_words(const uint32_t m[16], uint32_t block_len, uint8_t out[32]) {
    const uint32_t IV[8] = {
        0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au,
        0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u
    };
    const uint8_t S[7][16] = {
        {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15},
        {2,6,3,10,7,0,4,13,1,11,12,5,9,14,15,8},
        {3,4,10,12,13,2,7,14,6,5,9,0,11,15,8,1},
        {10,7,12,9,14,3,13,15,4,0,11,2,5,8,1,6},
        {12,13,9,11,15,10,14,8,7,2,5,3,0,1,6,4},
        {9,14,11,5,8,12,15,1,13,3,0,10,2,6,4,7},
        {11,15,5,0,1,9,8,6,14,10,2,12,3,4,7,13},
    };

    uint32_t v[16];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        v[i] = IV[i];
        v[i + 8] = IV[i];
    }
    v[12] = 0;
    v[13] = 0;
    v[14] = block_len;
    v[15] = 11u; // CHUNK_START | CHUNK_END | ROOT

    #pragma unroll
    for (int r = 0; r < 7; r++) {
        const uint8_t *s = S[r];
        gmix(v[0], v[4], v[8],  v[12], m[s[0]],  m[s[1]]);
        gmix(v[1], v[5], v[9],  v[13], m[s[2]],  m[s[3]]);
        gmix(v[2], v[6], v[10], v[14], m[s[4]],  m[s[5]]);
        gmix(v[3], v[7], v[11], v[15], m[s[6]],  m[s[7]]);
        gmix(v[0], v[5], v[10], v[15], m[s[8]],  m[s[9]]);
        gmix(v[1], v[6], v[11], v[12], m[s[10]], m[s[11]]);
        gmix(v[2], v[7], v[8],  v[13], m[s[12]], m[s[13]]);
        gmix(v[3], v[4], v[9],  v[14], m[s[14]], m[s[15]]);
    }

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        store32(out + i * 4, v[i] ^ v[i + 8]);
    }
}

__device__ void blake3_40(const uint8_t midstate[32], uint64_t nonce, uint8_t out[32]) {
    uint32_t m[16] = {0};
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        m[i] = load32(midstate + i * 4);
    }
    m[8] = (uint32_t)nonce;
    m[9] = (uint32_t)(nonce >> 32);
    blake3_oneblock_words(m, 40, out);
}

__device__ void blake3_32_inplace(uint8_t x[32]) {
    uint32_t m[16] = {0};
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        m[i] = load32(x + i * 4);
    }
    blake3_oneblock_words(m, 32, x);
}

__device__ __forceinline__ bool hash_less_than(const uint8_t h[32], const uint8_t target[32]) {
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        if (h[i] < target[i]) return true;
        if (h[i] > target[i]) return false;
    }
    return false;
}

__global__ void mine_kernel(
    const uint8_t *midstate,
    const uint8_t *target,
    uint64_t base,
    uint64_t n,
    uint32_t iterations,
    Candidate *cand) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = (uint64_t)gridDim.x * blockDim.x;

    for (uint64_t off = gid; off < n; off += stride) {
        uint64_t nonce = base + off;
        uint8_t h[32];
        blake3_40(midstate, nonce, h);
        for (uint32_t i = 0; i < iterations; i++) {
            blake3_32_inplace(h);
        }
        if (hash_less_than(h, target)) {
            uint32_t idx = atomicAdd(&cand->count, 1u);
            if (idx < cand->cap && idx < MAX_CANDIDATES) {
                cand->nonce[idx] = nonce;
                #pragma unroll
                for (int j = 0; j < 32; j++) cand->hash[idx][j] = h[j];
            }
        }
    }
}

static void usage(const char *argv0) {
    fprintf(stderr,
        "Usage: %s -o stratum+tcp://host:port -a ADDRESS [-w worker] [-d device]\n"
        "          [--blocks N] [--threads N] [--batch N] [--iters N]\n"
        "          [--no-dashboard]\n\n"
        "Defaults: all GPUs, --blocks 4096 --threads 128 --batch 524288 --iters 1000000\n",
        argv0);
}

static bool parse_args(int argc, char **argv, Options &o, uint32_t &iters) {
    iters = 1000000;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto need = [&](const char *name) -> const char * {
            if (i + 1 >= argc) {
                fprintf(stderr, "%s needs a value\n", name);
                exit(2);
            }
            return argv[++i];
        };
        if (a == "-o" || a == "--pool-url") o.pool = need(a.c_str());
        else if (a == "-a" || a == "--address" || a == "--payout-address") o.address = need(a.c_str());
        else if (a == "-w" || a == "--worker") o.worker = need(a.c_str());
        else if (a == "-d" || a == "--device") o.device = atoi(need(a.c_str()));
        else if (a == "--blocks") o.blocks = atoi(need(a.c_str()));
        else if (a == "--threads") o.threads = atoi(need(a.c_str()));
        else if (a == "--batch") o.batch = strtoull(need(a.c_str()), nullptr, 10);
        else if (a == "--iters") iters = (uint32_t)strtoul(need(a.c_str()), nullptr, 10);
        else if (a == "--seed") o.seed = strtoull(need(a.c_str()), nullptr, 10);
        else if (a == "--lane-index") o.lane_index = atoi(need(a.c_str()));
        else if (a == "--lane-count") o.lane_count = atoi(need(a.c_str()));
        else if (a == "--stats-dir") o.stats_dir = need(a.c_str());
        else if (a == "--no-dashboard") o.dashboard = false;
        else if (a == "--quiet") o.quiet = true;
        else if (a == "--child") o.child = true;
        else if (a == "-h" || a == "--help") { usage(argv[0]); exit(0); }
        else { fprintf(stderr, "unknown option: %s\n", a.c_str()); return false; }
    }
    return !o.address.empty();
}

static std::string format_elapsed(uint64_t seconds) {
    char buf[32];
    uint64_t h = seconds / 3600;
    uint64_t m = (seconds % 3600) / 60;
    uint64_t s = seconds % 60;
    snprintf(buf, sizeof(buf), "%02" PRIu64 ":%02" PRIu64 ":%02" PRIu64, h, m, s);
    return buf;
}

static std::string format_hashrate(double hps) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(2);
    if (hps >= 1e12) out << (hps / 1e12) << " TH/s";
    else if (hps >= 1e9) out << (hps / 1e9) << " GH/s";
    else if (hps >= 1e6) out << (hps / 1e6) << " MH/s";
    else if (hps >= 1e3) out << (hps / 1e3) << " kH/s";
    else out << hps << " H/s";
    return out.str();
}

static std::string short_text(const std::string &s, size_t n) {
    if (s.size() <= n) return s;
    if (n <= 3) return s.substr(0, n);
    return s.substr(0, n - 3) + "...";
}

struct FileStats {
    int gpu = -1;
    std::string name = "-";
    std::string worker = "-";
    std::string status = "starting";
    std::string job = "-";
    double hps = 0.0;
    double avg_hps = 0.0;
    uint64_t total = 0;
    uint64_t submitted = 0;
    uint64_t accepted = 0;
    uint64_t rejected = 0;
    uint64_t candidates = 0;
    uint64_t last_share_age = 0;
};

static std::string stats_path(const std::string &dir, int gpu) {
    return dir + "/gpu" + std::to_string(gpu) + ".stat";
}

static void write_stats_file(const Options &opt, const FileStats &s) {
    if (opt.stats_dir.empty()) return;
    std::string path = stats_path(opt.stats_dir, opt.device);
    std::string tmp = path + ".tmp";
    std::ofstream out(tmp);
    if (!out) return;
    out << "gpu=" << s.gpu << "\n";
    out << "name=" << s.name << "\n";
    out << "worker=" << s.worker << "\n";
    out << "status=" << s.status << "\n";
    out << "job=" << s.job << "\n";
    out << "hps=" << std::fixed << std::setprecision(3) << s.hps << "\n";
    out << "avg_hps=" << std::fixed << std::setprecision(3) << s.avg_hps << "\n";
    out << "total=" << s.total << "\n";
    out << "submitted=" << s.submitted << "\n";
    out << "accepted=" << s.accepted << "\n";
    out << "rejected=" << s.rejected << "\n";
    out << "candidates=" << s.candidates << "\n";
    out << "last_share_age=" << s.last_share_age << "\n";
    out.close();
    rename(tmp.c_str(), path.c_str());
}

static FileStats read_stats_file(const std::string &path) {
    FileStats s;
    std::ifstream in(path);
    std::string line;
    while (std::getline(in, line)) {
        size_t eq = line.find('=');
        if (eq == std::string::npos) continue;
        std::string k = line.substr(0, eq);
        std::string v = line.substr(eq + 1);
        if (k == "gpu") s.gpu = atoi(v.c_str());
        else if (k == "name") s.name = v;
        else if (k == "worker") s.worker = v;
        else if (k == "status") s.status = v;
        else if (k == "job") s.job = v;
        else if (k == "hps") s.hps = atof(v.c_str());
        else if (k == "avg_hps") s.avg_hps = atof(v.c_str());
        else if (k == "total") s.total = strtoull(v.c_str(), nullptr, 10);
        else if (k == "submitted") s.submitted = strtoull(v.c_str(), nullptr, 10);
        else if (k == "accepted") s.accepted = strtoull(v.c_str(), nullptr, 10);
        else if (k == "rejected") s.rejected = strtoull(v.c_str(), nullptr, 10);
        else if (k == "candidates") s.candidates = strtoull(v.c_str(), nullptr, 10);
        else if (k == "last_share_age") s.last_share_age = strtoull(v.c_str(), nullptr, 10);
    }
    return s;
}

static void render_dashboard(
    const Options &opt,
    const std::string &stats_dir,
    int device_count,
    const std::chrono::steady_clock::time_point &started) {
    std::vector<FileStats> snap;
    snap.reserve(device_count);
    for (int i = 0; i < device_count; i++) {
        FileStats s = read_stats_file(stats_path(stats_dir, i));
        if (s.gpu < 0) s.gpu = i;
        snap.push_back(s);
    }

    double total_hps = 0.0;
    uint64_t total_hashes = 0, submitted = 0, accepted = 0, rejected = 0, candidates = 0;
    for (const auto &s : snap) {
        total_hps += s.hps;
        total_hashes += s.total;
        submitted += s.submitted;
        accepted += s.accepted;
        rejected += s.rejected;
        candidates += s.candidates;
    }

    auto now = std::chrono::steady_clock::now();
    uint64_t elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - started).count();
    double avg_hps = elapsed > 0 ? (double)total_hashes / (double)elapsed : 0.0;

    std::ostringstream out;
    out << "\033[H\033[2J";
    out << "midstate-cuda-miner - NVIDIA CUDA Stratum miner\n";
    out << "Pool: " << opt.pool << "    Address: " << short_text(opt.address, 24)
        << "    Worker: " << opt.worker << "\n";
    out << "Uptime: " << format_elapsed(elapsed)
        << "    Current: " << format_hashrate(total_hps)
        << "    Average: " << format_hashrate(avg_hps)
        << "    GPUs: " << device_count << "\n";
    out << "Shares: accepted " << accepted
        << " | rejected " << rejected
        << " | submitted " << submitted
        << " | candidates " << candidates << "\n\n";

    out << std::left
        << std::setw(5) << "GPU"
        << std::setw(27) << "Device"
        << std::setw(15) << "Current"
        << std::setw(15) << "Average"
        << std::setw(10) << "Acc/Rej"
        << std::setw(11) << "Submitted"
        << std::setw(12) << "Job"
        << "Status\n";
    out << std::string(103, '-') << "\n";

    for (const auto &s : snap) {
        std::string accrej = std::to_string(s.accepted) + "/" + std::to_string(s.rejected);
        out << std::left
            << std::setw(5) << ("#" + std::to_string(s.gpu))
            << std::setw(27) << short_text(s.name, 26)
            << std::setw(15) << format_hashrate(s.hps)
            << std::setw(15) << format_hashrate(s.avg_hps)
            << std::setw(10) << accrej
            << std::setw(11) << s.submitted
            << std::setw(12) << short_text(s.job, 11)
            << short_text(s.status, 24) << "\n";
    }

    std::cout << out.str() << std::flush;
}

static bool parse_pool_url(const std::string &url, std::string &host, std::string &port) {
    std::string s = url;
    const std::string pfx = "stratum+tcp://";
    if (s.rfind(pfx, 0) == 0) s = s.substr(pfx.size());
    size_t colon = s.rfind(':');
    if (colon == std::string::npos) return false;
    host = s.substr(0, colon);
    port = s.substr(colon + 1);
    return !host.empty() && !port.empty();
}

static int connect_tcp(const std::string &host, const std::string &port) {
    struct addrinfo hints;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    struct addrinfo *res = nullptr;
    int rc = getaddrinfo(host.c_str(), port.c_str(), &hints, &res);
    if (rc != 0) {
        fprintf(stderr, "getaddrinfo: %s\n", gai_strerror(rc));
        return -1;
    }
    int fd = -1;
    for (auto *p = res; p; p = p->ai_next) {
        fd = socket(p->ai_family, p->ai_socktype, p->ai_protocol);
        if (fd < 0) continue;
        if (connect(fd, p->ai_addr, p->ai_addrlen) == 0) break;
        close(fd);
        fd = -1;
    }
    freeaddrinfo(res);
    return fd;
}

static bool send_all(int fd, const std::string &s) {
    const char *p = s.c_str();
    size_t left = s.size();
    while (left) {
        ssize_t n = send(fd, p, left, 0);
        if (n <= 0) return false;
        p += n;
        left -= (size_t)n;
    }
    return true;
}

static bool read_line(int fd, std::string &line) {
    line.clear();
    char c;
    while (true) {
        ssize_t n = recv(fd, &c, 1, 0);
        if (n <= 0) return false;
        if (c == '\n') return true;
        if (c != '\r') line.push_back(c);
    }
}

static int hexval(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

static bool hex_to_32(const std::string &hex, uint8_t out[32]) {
    if (hex.size() != 64) return false;
    for (int i = 0; i < 32; i++) {
        int hi = hexval(hex[i * 2]);
        int lo = hexval(hex[i * 2 + 1]);
        if (hi < 0 || lo < 0) return false;
        out[i] = (uint8_t)((hi << 4) | lo);
    }
    return true;
}

static std::string hash_hex(const uint8_t h[32]) {
    static const char *digits = "0123456789abcdef";
    std::string out;
    out.resize(64);
    for (int i = 0; i < 32; i++) {
        out[i * 2] = digits[h[i] >> 4];
        out[i * 2 + 1] = digits[h[i] & 15];
    }
    return out;
}

static bool parse_notify(const std::string &line, uint64_t &job_id, uint8_t midstate[32]) {
    if (line.find("mining.notify") == std::string::npos) return false;
    std::regex re("\"params\"\\s*:\\s*\\[\\s*([0-9]+)\\s*,\\s*\"([0-9a-fA-F]{64})\"");
    std::smatch m;
    if (!std::regex_search(line, m, re)) return false;
    job_id = strtoull(m[1].str().c_str(), nullptr, 10);
    return hex_to_32(m[2].str(), midstate);
}

static bool is_accepted(const std::string &line) {
    return line.find("\"id\":2") != std::string::npos && line.find("\"result\":true") != std::string::npos;
}

static bool is_rejected(const std::string &line) {
    return line.find("\"id\":2") != std::string::npos && line.find("\"result\":false") != std::string::npos;
}

static bool has_non_null_error(const std::string &line) {
    std::regex re("\"error\"\\s*:\\s*(null|\"[^\"]*\"|\\[[^\\]]*\\]|\\{[^\\}]*\\})");
    std::smatch m;
    if (!std::regex_search(line, m, re)) return false;
    return m[1].str() != "null";
}

int main(int argc, char **argv) {
    signal(SIGINT, on_sigint);
    signal(SIGTERM, on_sigint);

    Options opt;
    uint32_t iterations;
    if (!parse_args(argc, argv, opt, iterations)) {
        usage(argv[0]);
        return 2;
    }

    if (opt.device < 0) {
        int device_count = 0;
        CUDA_CHECK(cudaGetDeviceCount(&device_count));
        if (device_count <= 0) {
            fprintf(stderr, "no CUDA GPUs found\n");
            return 1;
        }
        uint64_t auto_seed = opt.seed;
        if (auto_seed == 0) {
            auto_seed = ((uint64_t)time(nullptr) << 32) ^ (uint64_t)getpid();
        }
        std::string stats_dir = opt.stats_dir;
        if (stats_dir.empty()) {
            stats_dir = "/tmp/midstate-cuda-miner-" + std::to_string((long long)getpid());
        }
        mkdir(stats_dir.c_str(), 0700);
        fprintf(stderr, "auto mode: launching %d GPU worker(s), seed=%" PRIu64 "\n", device_count, auto_seed);

        std::vector<pid_t> children;
        auto dashboard_started = std::chrono::steady_clock::now();
        for (int dev = 0; dev < device_count; dev++) {
            pid_t pid = fork();
            if (pid < 0) {
                perror("fork");
                g_stop = 1;
                break;
            }
            if (pid == 0) {
                std::vector<std::string> args;
                for (int i = 0; i < argc; i++) args.emplace_back(argv[i]);
                args.emplace_back("-d");
                args.emplace_back(std::to_string(dev));
                args.emplace_back("-w");
                args.emplace_back(opt.worker + "-gpu" + std::to_string(dev));
                args.emplace_back("--seed");
                args.emplace_back(std::to_string(auto_seed));
                args.emplace_back("--lane-index");
                args.emplace_back(std::to_string(dev));
                args.emplace_back("--lane-count");
                args.emplace_back(std::to_string(device_count));
                args.emplace_back("--stats-dir");
                args.emplace_back(stats_dir);
                args.emplace_back("--child");
                if (opt.dashboard) {
                    args.emplace_back("--quiet");
                }

                std::vector<char *> cargs;
                cargs.reserve(args.size() + 1);
                for (auto &s : args) cargs.push_back(const_cast<char *>(s.c_str()));
                cargs.push_back(nullptr);
                execv(argv[0], cargs.data());
                perror("execv");
                _exit(127);
            }
            children.push_back(pid);
        }

        int rc = 0;
        if (opt.dashboard) {
            std::cout << "\033[?1049h\033[?25l" << std::flush;
        }
        while (!children.empty()) {
            if (g_stop) {
                for (pid_t pid : children) {
                    kill(pid, SIGTERM);
                }
            }

            for (auto it = children.begin(); it != children.end();) {
                int status = 0;
                pid_t done = waitpid(*it, &status, WNOHANG);
                if (done == *it) {
                    if (status != 0) rc = 1;
                    it = children.erase(it);
                } else {
                    ++it;
                }
            }

            if (!children.empty()) {
                if (opt.dashboard) {
                    render_dashboard(opt, stats_dir, device_count, dashboard_started);
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
            }
        }
        if (opt.dashboard) {
            render_dashboard(opt, stats_dir, device_count, dashboard_started);
            std::cout << "\033[?25h\033[?1049l" << std::flush;
        }
        return rc;
    }

    CUDA_CHECK(cudaSetDevice(opt.device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, opt.device));
    if (!opt.quiet) {
        fprintf(stderr, "CUDA device %d: %s, cc %d.%d\n", opt.device, prop.name, prop.major, prop.minor);
    }

    uint8_t share_target_host[32];
    memset(share_target_host, 0xff, sizeof(share_target_host));
    share_target_host[0] = 0x00;
    share_target_host[1] = 0x0f;

    uint8_t *d_midstate = nullptr;
    uint8_t *d_target = nullptr;
    Candidate *d_cand = nullptr;
    Candidate h_cand;
    CUDA_CHECK(cudaMalloc(&d_midstate, 32));
    CUDA_CHECK(cudaMalloc(&d_target, 32));
    CUDA_CHECK(cudaMalloc(&d_cand, sizeof(Candidate)));
    CUDA_CHECK(cudaMemcpy(d_target, share_target_host, 32, cudaMemcpyHostToDevice));

    std::string host, port;
    if (!parse_pool_url(opt.pool, host, port)) {
        fprintf(stderr, "invalid pool URL: %s\n", opt.pool.c_str());
        return 2;
    }

    if (opt.lane_count < 1) opt.lane_count = 1;
    if (opt.lane_index < 0 || opt.lane_index >= opt.lane_count) opt.lane_index = 0;
    uint64_t base_seed = opt.seed;
    if (base_seed == 0) {
        base_seed = ((uint64_t)time(nullptr) << 32) ^ (uint64_t)getpid();
    }
    uint64_t base = base_seed + (uint64_t)opt.lane_index * opt.batch;
    uint64_t job_id = 0;
    uint8_t job_midstate[32] = {0};
    bool have_job = false;
    uint64_t accepted = 0, rejected = 0, checked = 0, total_hashes = 0, submitted = 0, candidates = 0;
    double current_hps = 0.0;
    auto t0 = std::chrono::steady_clock::now();
    auto started = t0;
    auto last_share = started;
    bool has_share = false;
    FileStats fs;
    fs.gpu = opt.device;
    fs.name = prop.name;
    fs.worker = opt.worker;
    fs.status = "starting";
    write_stats_file(opt, fs);

    while (!g_stop) {
        int fd = connect_tcp(host, port);
        if (fd < 0) {
            fs.status = "connect failed";
            write_stats_file(opt, fs);
            if (!opt.quiet) fprintf(stderr, "connect failed; retrying in 5s\n");
            std::this_thread::sleep_for(std::chrono::seconds(5));
            continue;
        }
        fs.status = "connected";
        write_stats_file(opt, fs);
        if (!opt.quiet) fprintf(stderr, "connected to %s:%s\n", host.c_str(), port.c_str());

        std::string sub = "{\"id\":1,\"method\":\"mining.subscribe\",\"params\":[]}\n";
        std::string auth = "{\"id\":1,\"method\":\"mining.authorize\",\"params\":[\"" +
            opt.address + "\",\"" + opt.worker + "\"]}\n";
        send_all(fd, sub);
        send_all(fd, auth);
        auto connected_at = std::chrono::steady_clock::now();

        while (!g_stop) {
            fd_set rfds;
            FD_ZERO(&rfds);
            FD_SET(fd, &rfds);
            struct timeval tv;
            tv.tv_sec = have_job ? 0 : 10;
            tv.tv_usec = have_job ? 1000 : 0;
            int sr = select(fd + 1, &rfds, nullptr, nullptr, &tv);
            if (sr < 0 && errno != EINTR) break;
            if (sr > 0 && FD_ISSET(fd, &rfds)) {
                std::string line;
                if (!read_line(fd, line)) break;
                uint64_t jid;
                uint8_t ms[32];
                if (parse_notify(line, jid, ms)) {
                    job_id = jid;
                    memcpy(job_midstate, ms, 32);
                    CUDA_CHECK(cudaMemcpy(d_midstate, job_midstate, 32, cudaMemcpyHostToDevice));
                    have_job = true;
                    fs.job = std::to_string(job_id);
                    fs.status = "mining";
                    write_stats_file(opt, fs);
                    if (!opt.quiet) fprintf(stderr, "new job %" PRIu64 " midstate=%s\n", job_id, hash_hex(job_midstate).c_str());
                } else if (line.find("\"id\":1") != std::string::npos &&
                           line.find("\"result\":true") != std::string::npos) {
                    fs.status = "handshake ok";
                    write_stats_file(opt, fs);
                    if (!opt.quiet) fprintf(stderr, "pool handshake ok\n");
                } else if (is_accepted(line)) {
                    accepted++;
                    has_share = true;
                    last_share = std::chrono::steady_clock::now();
                    fs.accepted = accepted;
                    fs.status = "share accepted";
                    write_stats_file(opt, fs);
                    if (!opt.quiet) fprintf(stderr, "share accepted (%" PRIu64 " acc / %" PRIu64 " rej)\n", accepted, rejected);
                } else if (is_rejected(line) || has_non_null_error(line)) {
                    rejected++;
                    fs.rejected = rejected;
                    fs.status = "share rejected";
                    write_stats_file(opt, fs);
                    if (!opt.quiet) fprintf(stderr, "share rejected: %s\n", line.c_str());
                }
            }

            if (!have_job) {
                auto now = std::chrono::steady_clock::now();
                if (std::chrono::duration_cast<std::chrono::seconds>(now - connected_at).count() > 15) {
                    fs.status = "no job reconnect";
                    write_stats_file(opt, fs);
                    if (!opt.quiet) fprintf(stderr, "no mining.notify after handshake; reconnecting\n");
                    break;
                }
                continue;
            }

            h_cand = {};
            h_cand.cap = MAX_CANDIDATES;
            CUDA_CHECK(cudaMemcpy(d_cand, &h_cand, sizeof(h_cand), cudaMemcpyHostToDevice));
            mine_kernel<<<opt.blocks, opt.threads>>>(d_midstate, d_target, base, opt.batch, iterations, d_cand);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            checked += opt.batch;
            total_hashes += opt.batch;
            CUDA_CHECK(cudaMemcpy(&h_cand, d_cand, sizeof(h_cand), cudaMemcpyDeviceToHost));

            auto now = std::chrono::steady_clock::now();
            double secs = std::chrono::duration<double>(now - t0).count();
            if (secs >= 5.0) {
                current_hps = checked / secs;
                double avg_hps = std::chrono::duration<double>(now - started).count() > 0.0
                    ? (double)total_hashes / std::chrono::duration<double>(now - started).count()
                    : 0.0;
                fs.hps = current_hps;
                fs.avg_hps = avg_hps;
                fs.total = total_hashes;
                fs.submitted = submitted;
                fs.accepted = accepted;
                fs.rejected = rejected;
                fs.candidates = candidates;
                fs.last_share_age = has_share
                    ? (uint64_t)std::chrono::duration_cast<std::chrono::seconds>(now - last_share).count()
                    : 0;
                if (fs.status == "share accepted" || fs.status == "share rejected") {
                    fs.status = "mining";
                }
                write_stats_file(opt, fs);
                if (!opt.quiet) {
                    fprintf(stderr, "hashrate: %.2f H/s checked=%" PRIu64 " worker=%s\n",
                        current_hps, checked, opt.worker.c_str());
                }
                t0 = now;
                checked = 0;
            }

            uint32_t found_count = h_cand.count;
            if (found_count > MAX_CANDIDATES) found_count = MAX_CANDIDATES;
            if (found_count > 0) {
                candidates += found_count;
                fs.candidates = candidates;
                fs.status = "candidate";
                write_stats_file(opt, fs);
                if (!opt.quiet) fprintf(stderr, "candidates=%u first_nonce=%" PRIu64 " first_hash=%s\n",
                    found_count, h_cand.nonce[0], hash_hex(h_cand.hash[0]).c_str());
                bool submit_ok = true;
                for (uint32_t i = 0; i < found_count; i++) {
                    char buf[512];
                    snprintf(buf, sizeof(buf),
                        "{\"id\":2,\"method\":\"mining.submit\",\"params\":[\"%s\",%" PRIu64 ",%" PRIu64 "]}\n",
                        opt.address.c_str(), job_id, h_cand.nonce[i]);
                    submitted++;
                    fs.submitted = submitted;
                    write_stats_file(opt, fs);
                    if (!send_all(fd, buf)) {
                        submit_ok = false;
                        break;
                    }
                }
                if (!submit_ok) break;
            }
            base += opt.batch * (uint64_t)opt.lane_count;
        }
        close(fd);
        have_job = false;
        fs.status = "disconnected";
        write_stats_file(opt, fs);
        if (!g_stop) {
            if (!opt.quiet) fprintf(stderr, "disconnected; retrying in 5s\n");
            std::this_thread::sleep_for(std::chrono::seconds(5));
        }
    }

    cudaFree(d_midstate);
    cudaFree(d_target);
    cudaFree(d_cand);
    return 0;
}
