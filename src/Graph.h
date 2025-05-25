// Graph.h
#pragma once
#include <vector>
#include <unordered_map>
#include "Node.h"
#include "Tensor.h"

class Graph {
public:
  std::vector<Node> nodes;
  // tensor store: name → actual tensor data
  std::unordered_map<std::string, Tensor<>> tensors;

  // Add node and wire parent/child via matching names
  void add_node(const Node &n) { nodes.push_back(n); }
};
