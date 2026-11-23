#include <chrono>
#include <vector>
#include "configuration_mis.h"
#include "ils/online_ils.h"

namespace {
using Clock = std::chrono::steady_clock;
using Emit = void (*)(int, const int *);
thread_local Emit receive;
thread_local Clock::time_point stop_search;
thread_local int archived_size;

void publish(graph_access &graph, mis_permutation_online *permutation) {
    const int size = permutation->get_solution_size();
    if (size <= archived_size) return;
    std::vector<int> nodes;
    nodes.reserve(size);
    for (NodeID vertex = 0; vertex < graph.number_of_nodes(); ++vertex) {
        if (permutation->is_solution_node(vertex)) nodes.push_back(vertex);
    }
    receive(static_cast<int>(nodes.size()), nodes.data());
    archived_size = size;
}
}

extern "C" void run_online_mis(int n, int *row, int *col, unsigned seed, double seconds, Emit emit) {
    stop_search = Clock::now() + std::chrono::duration_cast<Clock::duration>(
        std::chrono::duration<double>(seconds));
    archived_size = -1;
    receive = emit;
    MISConfig config;
    configuration_mis defaults;
    defaults.standard(config);
    config.seed = seed;
    graph_access graph;
    graph.build_from_metis(n, row, col);
    online_ils algorithm;
    algorithm.perform_ils(config, graph, config.ils_iterations);
}
