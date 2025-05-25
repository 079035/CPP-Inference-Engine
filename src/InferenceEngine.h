#pragma once
#include "Graph.h"
#include "GraphUtils.h"
#include "operators.h"

class InferenceEngine {
  Graph graph_;
  std::vector<size_t> topo_order_;

public:
  InferenceEngine(const onnx::ModelProto &model) {
    build_graph(model);
    topo_order_ = topo_sort(graph_);
  }

  // map op_type → function pointer
  using OpFn = std::function<Tensor<>(const std::vector<Tensor<>>&,
                                      const Node&)>;
  std::unordered_map<std::string, OpFn> op_registry;

  void build_graph(const onnx::ModelProto &model);
  Tensor<> infer(const std::unordered_map<std::string, Tensor<>>& inputs);
};
