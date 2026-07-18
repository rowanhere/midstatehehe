#include "../src/stratum_reader.hpp"

#include <sys/socket.h>
#include <unistd.h>

#include <cassert>
#include <chrono>
#include <string>

static void send_text(int fd, const std::string &text) {
    const char *data = text.data();
    size_t remaining = text.size();
    while (remaining > 0) {
        ssize_t sent = send(fd, data, remaining, 0);
        assert(sent > 0);
        data += sent;
        remaining -= static_cast<size_t>(sent);
    }
}

int main() {
    int sockets[2];
    assert(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0);

    StratumReader reader(sockets[0]);
    std::string burst;
    for (int id = 1000; id < 1100; ++id) {
        burst += "{\"id\":" + std::to_string(id) + ",\"result\":true}\n";
    }
    send_text(sockets[1], burst);

    std::string line;
    for (int id = 1000; id < 1100; ++id) {
        assert(reader.pop(line, std::chrono::seconds(1)));
        assert(line == "{\"id\":" + std::to_string(id) + ",\"result\":true}");
    }

    send_text(sockets[1], "{\"id\":1100,");
    assert(!reader.pop(line, std::chrono::milliseconds(20)));
    send_text(sockets[1], "\"result\":true}\r\n{\"id\":1101,\"result\":false}\n");
    assert(reader.pop(line, std::chrono::seconds(1)));
    assert(line == "{\"id\":1100,\"result\":true}");
    assert(reader.pop(line, std::chrono::seconds(1)));
    assert(line == "{\"id\":1101,\"result\":false}");

    close(sockets[1]);
    for (int i = 0; i < 100 && reader.is_alive(); ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    assert(!reader.is_alive());
    reader.stop();
    close(sockets[0]);
    return 0;
}
