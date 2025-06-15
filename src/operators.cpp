// operators.cpp
#include "operators.h"
#ifdef USE_CUDA
#include "operators_cuda.h"
#endif
#include <algorithm>

// Flatten
Tensor<> op_flatten(const Tensor<> &in) {
  Tensor<> out = in;
  out.flatten_to_2d();
  return out;
}

// Add
Tensor<> op_add(const Tensor<> &A, const Tensor<> &B) {
  assert(A.data.size() == B.data.size());
  Tensor<> out(A.shape);
  for (size_t i = 0; i < A.data.size(); ++i)
    out.data[i] = A.data[i] + B.data[i];
  return out;
}

// ReLU
Tensor<> op_relu(const Tensor<> &in) {
  Tensor<> out(in.shape);
  std::transform(in.data.begin(), in.data.end(), out.data.begin(),
                  [](float x){ return x > 0 ? x : 0; });
  return out;
}

// Gemm: C = α·A·B + β·bias
Tensor<> op_gemm(const Tensor<> &A, const Tensor<> &B,
                  const Tensor<> &bias, bool transA, bool transB,
                  float alpha, float beta) {
#ifdef USE_CUDA
  // For CUDA path, make copies of inputs and move them to device
  Tensor<> A_device = A;
  Tensor<> B_device = B;
  Tensor<> bias_device = bias;
  
  A_device.toDevice();
  B_device.toDevice();
  bias_device.toDevice();
  
  int64_t M = transA ? A.shape[1] : A.shape[0];
  int64_t N = transB ? B.shape[0] : B.shape[1];
  Tensor<> C({M, N});
  C.toDevice();
  
  cuda_gemm(A_device, B_device, bias_device, C, transA, transB, alpha, beta);
  
  // Move result back to host for consistency with CPU path
  C.toHost();
  return C;
#else
  // CPU fallback
  int64_t M = transA ? A.shape[1] : A.shape[0];
  int64_t K = transA ? A.shape[0] : A.shape[1];
  int64_t N = transB ? B.shape[0] : B.shape[1];
  Tensor<> C({M, N});
  for (int64_t m = 0; m < M; ++m) {
    for (int64_t n = 0; n < N; ++n) {
      float sum = 0;
      for (int64_t k = 0; k < K; ++k) {
        float a = transA ? A.data[k*M + m] : A.data[m*K + k];
        float b = transB ? B.data[n*K + k] : B.data[k*N + n];
        sum += a * b;
      }
      C.data[m*N + n] = alpha * sum + beta * bias.data[n];
    }
  }
  return C;
#endif
}
