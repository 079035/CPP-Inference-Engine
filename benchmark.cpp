#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <filesystem>
#include "ONNXModelLoader.h"
#include "InferenceEngine.h"
#include "Tensor.h"

namespace fs = std::filesystem;

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
        tensor.data[i] = static_cast<float>(pixel) / 255.0f;
    }
    return tensor;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " model.onnx inputs_folder\n";
        return 1;
    }
    const std::string model_path = argv[1];
    const std::string input_dir = argv[2];
    try {
        auto model = ONNXModelLoader::load(model_path);
        InferenceEngine engine(model);
        std::vector<int64_t> input_shape = {1, 1, 28, 28};
        std::string input_name = "onnx::Flatten_0";
        std::vector<std::string> input_files;
        for (const auto& entry : fs::directory_iterator(input_dir)) {
            if (entry.is_regular_file() && entry.path().extension() == ".ubyte") {
                input_files.push_back(entry.path().string());
            }
        }
        if (input_files.empty()) {
            std::cerr << "No .ubyte files found in " << input_dir << std::endl;
            return 1;
        }
        const int num_loops = 100;
        std::vector<double> times;
        for (int loop = 0; loop < num_loops; ++loop) {
            for (const auto& input_path : input_files) {
                Tensor<float> input_tensor = loadUByteImage(input_path, input_shape);
                auto start = std::chrono::high_resolution_clock::now();
                auto output = engine.infer({{input_name, input_tensor}});
                auto end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double, std::milli> elapsed = end - start;
                times.push_back(elapsed.count());
            }
        }
        double total = std::accumulate(times.begin(), times.end(), 0.0);
        double avg = total / times.size();
        std::cout << "Ran " << input_files.size() << " inputs, " << num_loops << " times each (" << times.size() << " total runs)\n";
        std::cout << "Total inference time: " << total << " ms\n";
        std::cout << "Average inference time per run: " << avg << " ms\n";
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << '\n';
        return 1;
    }
    return 0;
} 