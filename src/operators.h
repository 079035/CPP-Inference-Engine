// operators.h
#pragma once
#include "Tensor.h"

Tensor<> op_flatten(const Tensor<> &in);
Tensor<> op_add(const Tensor<> &A, const Tensor<> &B);
Tensor<> op_relu(const Tensor<> &in);
Tensor<> op_gemm(const Tensor<> &A, const Tensor<> &B, const Tensor<> &bias,
                    bool transA=false, bool transB=false, float alpha=1.0f, float beta=1.0f);
