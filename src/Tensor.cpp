#include "Tensor.h"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <stdexcept>
#endif

// Only implement for float for now
template<>
Tensor<float>::~Tensor() {
#ifdef USE_CUDA
    if (device_data) {
        cudaFree(device_data);
        device_data = nullptr;
    }
#endif
}

template<>
void Tensor<float>::toDevice() {
#ifdef USE_CUDA
    if (device == DeviceType::CUDA) return; // Already on device
    size_t bytes = data.size() * sizeof(float);
    if (!device_data) {
        cudaError_t err = cudaMalloc(&device_data, bytes);
        if (err != cudaSuccess) throw std::runtime_error("cudaMalloc failed");
    }
    cudaError_t err = cudaMemcpy(device_data, data.data(), bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) throw std::runtime_error("cudaMemcpy H2D failed");
    device = DeviceType::CUDA;
#endif
}

template<>
void Tensor<float>::toHost() {
#ifdef USE_CUDA
    if (device == DeviceType::CPU) return; // Already on host
    size_t bytes = data.size() * sizeof(float);
    if (device_data) {
        cudaError_t err = cudaMemcpy(data.data(), device_data, bytes, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) throw std::runtime_error("cudaMemcpy D2H failed");
    }
    device = DeviceType::CPU;
#endif
}
