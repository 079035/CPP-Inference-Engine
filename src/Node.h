// Node.h
#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include "onnx-ml.pb.h"

struct Node {
  std::string                     op_type;       // e.g. "Gemm", "ReLU"
  std::vector<std::string>        input_names;   // tensor keys
  std::vector<std::string>        output_names;
  std::unordered_map<std::string, onnx::AttributeProto> attrs;
};
