#pragma once
#include "Tensor.h"

void cuda_gemm(const Tensor<float>& A, const Tensor<float>& B, const Tensor<float>& bias,
               Tensor<float>& C, bool transA, bool transB, float alpha, float beta); 