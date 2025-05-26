#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include "ONNXModelLoader.h"
#include "InferenceEngine.h"
#include "Tensor.h"

Tensor<float> loadUByteImage(const std::string& path, const std::vector<int64_t>& shape) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open image: " + path);
    }

    size_t total = 1;
    for (auto d : shape) total *= d;

    Tensor<float> tensor(shape);
    for (size_t i = 0; i < total; ++i) {
        uint8_t pixel;
        file.read(reinterpret_cast<char*>(&pixel), 1);
        tensor.data[i] = static_cast<float>(pixel) / 255.0f;  // Normalize to [0,1]
    }
    return tensor;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " models/mnist_net.onnx inputs/image_0.ubyte\n";
        return 1;
    }

    const std::string model_path = argv[1];
    const std::string image_path = argv[2];

    try {
        // Load ONNX model
        auto model = ONNXModelLoader::load(model_path);
        std::cout << "Model input(s): ";
        for (const auto& input : model.graph().input()) {
            std::cout << input.name() << " shape: ";
            for (const auto& d : input.type().tensor_type().shape().dim()) {
                std::cout << d.dim_value() << " ";
            }
            std::cout << std::endl;
        }

        std::cout << std::endl;
        InferenceEngine engine(model);

        // Shape depends on model — example assumes [1, 1, 28, 28] (e.g., MNIST)
        std::vector<int64_t> input_shape = {1, 1, 28, 28};
        Tensor<float> input_tensor = loadUByteImage(image_path, input_shape);

        // Input name (adjust based on your ONNX model's input)
        std::string input_name = "onnx::Flatten_0";

        auto output = engine.infer({{input_name, input_tensor}});
        
        std::cout << "Output shape: ";
        for(auto itr:output.shape){
            std::cout << itr << " ";
        }
        std::cout << std::endl << "Output data: ";
        for(auto itr:output.data){
            std::cout << itr << " ";
        }
        std::cout << std::endl;

        // Output handling: e.g., argmax
        auto max_it = std::max_element(output.data.begin(), output.data.end());
        std::cout << "Max element: " << *max_it << std::endl;
        int predicted_class = std::distance(output.data.begin(), max_it);
        std::cout << "Predicted class: " << predicted_class << std::endl;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << '\n';
        return 1;
    }

    return 0;
}
