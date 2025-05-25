#pragma once
#include <vector>
#include <queue>
#include <unordered_map>
#include "Graph.h"

inline std::vector<size_t> topo_sort(const Graph& graph) {
    std::unordered_map<size_t, int> indegree;
    std::unordered_map<std::string, std::vector<size_t>> consumers;

    for (size_t i = 0; i < graph.nodes.size(); ++i) {
        for (const auto& input : graph.nodes[i].input_names) {
            consumers[input].push_back(i);
        }
    }

    for (size_t i = 0; i < graph.nodes.size(); ++i) {
        indegree[i] = 0;
    }

    for (size_t i = 0; i < graph.nodes.size(); ++i) {
        for (const auto& output : graph.nodes[i].output_names) {
            for (auto c : consumers[output]) {
                ++indegree[c];
            }
        }
    }

    std::queue<size_t> q;
    for (auto& [i, deg] : indegree) {
        if (deg == 0) q.push(i);
    }

    std::vector<size_t> order;
    while (!q.empty()) {
        size_t u = q.front(); q.pop();
        order.push_back(u);
        for (const auto& output : graph.nodes[u].output_names) {
            for (auto c : consumers[output]) {
                if (--indegree[c] == 0)
                    q.push(c);
            }
        }
    }

    return order;
}
