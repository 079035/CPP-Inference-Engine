// ONNXModelLoader.h
#include <fstream>
#include "onnx-ml.pb.h"

class ONNXModelLoader {
public:
  static onnx::ModelProto load(const std::string &path) {
    onnx::ModelProto model;
    std::ifstream in(path, std::ios::binary);
    if (!model.ParseFromIstream(&in)) {
      throw std::runtime_error("Failed to parse ONNX model");
    }
    return model;
  }
};
