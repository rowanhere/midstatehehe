#include "../src/stratum_protocol.hpp"

#include <cassert>
#include <string>

int main() {
    const std::string hash(64, 'a');
    std::string target = "[0,0,0,57,68,3";
    for (int i = 6; i < 32; ++i) target += ",255";
    target += "]";
    const std::string padding(12000, 'x');
    const std::string notify =
        "{\"id\":null,\"method\":\"mining.notify\",\"params\":[1784368194,\"" + hash +
        "\",{\"padding\":\"" + padding + "\",\"target\":" + target + "}]}";

    uint64_t job_id = 0;
    uint8_t midstate[32] = {};
    uint8_t network_target[32] = {};
    assert(parse_notify(notify, job_id, midstate, network_target));
    assert(job_id == 1784368194);
    assert(midstate[0] == 0xaa && midstate[31] == 0xaa);
    assert(network_target[0] == 0);
    assert(network_target[3] == 57);
    assert(network_target[4] == 68);
    assert(network_target[5] == 3);
    assert(network_target[31] == 255);

    uint8_t block_hash[32] = {};
    block_hash[3] = 56;
    assert(protocol_hash_below_target(block_hash, network_target));
    block_hash[3] = 58;
    assert(!protocol_hash_below_target(block_hash, network_target));

    uint64_t response_id = 0;
    bool accepted = false;
    assert(parse_submit_response("{\"id\":1007,\"result\":true,\"error\":null}", response_id, accepted));
    assert(response_id == 1007 && accepted);
    assert(has_non_null_error("{\"id\":1008,\"result\":false,\"error\":\"Stale job\"}"));
    return 0;
}
