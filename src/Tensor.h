// Tensor.h
#pragma once
#include <vector>
#include <numeric>
#include <cassert>
#include <cstddef>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

// Device type for tensor data
enum class DeviceType {
    CPU,
    CUDA
};

template<typename T=float>
struct Tensor {
  std::vector<int64_t> shape;      // e.g. {1, 512, 28, 28}
  std::vector<T>       data;       // row-major (CPU)
  void*                device_data = nullptr; // CUDA device pointer
  DeviceType           device = DeviceType::CPU;

  Tensor() = default;
  Tensor(std::vector<int64_t> s)
    : shape(std::move(s)),
      data(std::accumulate(shape.begin(), shape.end(), 1LL, std::multiplies<int64_t>())) {}

  ~Tensor(); // Destructor to free device memory if needed

  // Flatten to 1D
  void flatten() {
    data.resize(data.size());
    shape = { static_cast<int64_t>(data.size()) };
  }

  void flatten_to_2d() {
    // Flatten all dimensions except the first (batch)
    assert(shape.size() >= 2);
    int64_t batch = shape[0];
    int64_t rest = 1;
    for (size_t i = 1; i < shape.size(); ++i)
        rest *= shape[i];
    shape = {batch, rest};
    // data vector remains the same
  }

  // access element (row-major)
  T& operator[](size_t idx) {
    assert(idx < data.size());
    return data[idx];
  }

  // CUDA-related methods (interface only)
  void toDevice();   // Copy data to device (GPU)
  void toHost();     // Copy data to host (CPU)
  bool isOnDevice() const { return device == DeviceType::CUDA; }
  bool isOnHost() const { return device == DeviceType::CPU; }
};
