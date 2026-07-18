#pragma once

#include <ctype.h>
#include <stdint.h>
#include <stdlib.h>

#include <regex>
#include <string>

inline int protocol_hexval(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

inline bool protocol_hex_to_32(const std::string &hex, uint8_t out[32]) {
    if (hex.size() != 64) return false;
    for (int i = 0; i < 32; ++i) {
        int hi = protocol_hexval(hex[i * 2]);
        int lo = protocol_hexval(hex[i * 2 + 1]);
        if (hi < 0 || lo < 0) return false;
        out[i] = static_cast<uint8_t>((hi << 4) | lo);
    }
    return true;
}

inline bool protocol_hash_below_target(const uint8_t hash[32], const uint8_t target[32]) {
    for (int i = 0; i < 32; ++i) {
        if (hash[i] < target[i]) return true;
        if (hash[i] > target[i]) return false;
    }
    return false;
}

inline bool parse_target_array(const std::string &line, uint8_t target[32]) {
    size_t field = line.find("\"target\"");
    if (field == std::string::npos) return false;
    size_t cursor = line.find('[', field);
    if (cursor == std::string::npos) return false;
    ++cursor;

    for (size_t i = 0; i < 32; ++i) {
        while (cursor < line.size() && isspace(static_cast<unsigned char>(line[cursor]))) ++cursor;
        if (cursor >= line.size() || line[cursor] < '0' || line[cursor] > '9') return false;
        unsigned value = 0;
        while (cursor < line.size() && line[cursor] >= '0' && line[cursor] <= '9') {
            value = value * 10 + static_cast<unsigned>(line[cursor++] - '0');
            if (value > 255) return false;
        }
        target[i] = static_cast<uint8_t>(value);
        while (cursor < line.size() && isspace(static_cast<unsigned char>(line[cursor]))) ++cursor;
        if (i < 31) {
            if (cursor >= line.size() || line[cursor] != ',') return false;
            ++cursor;
        }
    }
    while (cursor < line.size() && isspace(static_cast<unsigned char>(line[cursor]))) ++cursor;
    return cursor < line.size() && line[cursor] == ']';
}

inline bool parse_notify(
    const std::string &line,
    uint64_t &job_id,
    uint8_t midstate[32],
    uint8_t network_target[32]) {
    if (line.find("\"method\":\"mining.notify\"") == std::string::npos) return false;
    size_t params = line.find("\"params\"");
    if (params == std::string::npos) return false;
    size_t cursor = line.find('[', params);
    if (cursor == std::string::npos) return false;
    ++cursor;
    while (cursor < line.size() && isspace(static_cast<unsigned char>(line[cursor]))) ++cursor;
    size_t number_start = cursor;
    while (cursor < line.size() && line[cursor] >= '0' && line[cursor] <= '9') ++cursor;
    if (cursor == number_start) return false;
    job_id = strtoull(line.substr(number_start, cursor - number_start).c_str(), nullptr, 10);

    cursor = line.find('"', cursor);
    if (cursor == std::string::npos || cursor + 65 > line.size()) return false;
    if (!protocol_hex_to_32(line.substr(cursor + 1, 64), midstate)) return false;
    return parse_target_array(line, network_target);
}

inline bool parse_submit_response(const std::string &line, uint64_t &id, bool &accepted) {
    static const std::regex id_re("\"id\"\\s*:\\s*([0-9]+)");
    static const std::regex result_re("\"result\"\\s*:\\s*(true|false)");
    std::smatch id_match;
    std::smatch result_match;
    if (!std::regex_search(line, id_match, id_re)) return false;
    if (!std::regex_search(line, result_match, result_re)) return false;
    id = strtoull(id_match[1].str().c_str(), nullptr, 10);
    accepted = result_match[1].str() == "true";
    return true;
}

inline bool has_non_null_error(const std::string &line) {
    static const std::regex error_re("\"error\"\\s*:\\s*(null|\"[^\"]*\"|\\[[^\\]]*\\]|\\{[^\\}]*\\})");
    std::smatch match;
    if (!std::regex_search(line, match, error_re)) return false;
    return match[1].str() != "null";
}
