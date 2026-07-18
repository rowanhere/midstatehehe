#pragma once

#include <sys/socket.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <utility>

class StratumReader {
public:
    explicit StratumReader(int socket_fd) : fd(socket_fd), worker([this] { run(); }) {}

    StratumReader(const StratumReader &) = delete;
    StratumReader &operator=(const StratumReader &) = delete;

    ~StratumReader() {
        stop();
    }

    bool pop(std::string &line, std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mu);
        if (lines.empty() && alive.load(std::memory_order_acquire) && timeout.count() > 0) {
            cv.wait_for(lock, timeout, [this] {
                return !lines.empty() || !alive.load(std::memory_order_acquire);
            });
        }
        if (lines.empty()) return false;
        line = std::move(lines.front());
        lines.pop_front();
        return true;
    }

    bool is_alive() const {
        return alive.load(std::memory_order_acquire);
    }

    void stop() {
        if (!stopping.exchange(true, std::memory_order_acq_rel)) {
            shutdown(fd, SHUT_RDWR);
        }
        if (worker.joinable()) worker.join();
    }

private:
    void run() {
        std::string buffer;
        buffer.reserve(65536);

        while (!stopping.load(std::memory_order_acquire)) {
            size_t newline;
            while ((newline = buffer.find('\n')) != std::string::npos) {
                std::string line = buffer.substr(0, newline);
                if (!line.empty() && line.back() == '\r') line.pop_back();
                buffer.erase(0, newline + 1);
                {
                    std::lock_guard<std::mutex> lock(mu);
                    lines.push_back(std::move(line));
                }
                cv.notify_one();
            }

            char incoming[8192];
            ssize_t count = recv(fd, incoming, sizeof(incoming), 0);
            if (count <= 0) break;
            buffer.append(incoming, static_cast<size_t>(count));
        }

        alive.store(false, std::memory_order_release);
        cv.notify_all();
    }

    int fd;
    std::mutex mu;
    std::condition_variable cv;
    std::deque<std::string> lines;
    std::atomic<bool> alive{true};
    std::atomic<bool> stopping{false};
    std::thread worker;
};
