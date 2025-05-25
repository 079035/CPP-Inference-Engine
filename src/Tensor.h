// Tensor.h
#pragma once
#include <vector>
#include <numeric>
#include <cassert>

template<typename T=float>
struct Tensor {
  std::vector<int64_t> shape;      // e.g. {1, 512, 28, 28}
  std::vector<T>       data;       // row-major

  Tensor() = default;
  Tensor(std::vector<int64_t> s)
    : shape(std::move(s)),
      data(std::accumulate(shape.begin(), shape.end(), 1LL, std::multiplies<int64_t>())) {}

  // Flatten to 1D
  void flatten() {
    data.resize(data.size());
    shape = { static_cast<int64_t>(data.size()) };
  }

  // access element (row-major)
  T& operator[](size_t idx) {
    assert(idx < data.size());
    return data[idx];
  }
};
