#include "Tensor.h"
#include <cuda_runtime.h>
#include <stdexcept>

// CUDA kernel for GEMM: C = alpha * A * B + beta * bias
__global__ void gemm_kernel(const float* A, const float* B, const float* bias, float* C,
                            int M, int N, int K, float alpha, float beta) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = alpha * sum + beta * bias[col];
    }
}

// Launcher for CUDA GEMM
void cuda_gemm(const Tensor<float>& A, const Tensor<float>& B, const Tensor<float>& bias,
                Tensor<float>& C, bool transA, bool transB, float alpha, float beta) {
    // Only support non-transposed for now
    if (transA || transB) throw std::runtime_error("CUDA GEMM: transposed not supported yet");
    int M = A.shape[0];
    int K = A.shape[1];
    int N = B.shape[1];
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    gemm_kernel<<<grid, block>>>(
        static_cast<const float*>(A.device_data),
        static_cast<const float*>(B.device_data),
        static_cast<const float*>(bias.device_data),
        static_cast<float*>(C.device_data),
        M, N, K, alpha, beta
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) throw std::runtime_error("CUDA kernel launch failed");
}
