#ifndef OPS_H
#define OPS_H

#include "tensor.h"

//kernels
__global__ void matmul_kernel(float* M, float* N, float* P, int j, int k, int l, unsigned int Mds_sz);

// Forward operations on tensors
Tensor* tensor_add(Tensor* a, Tensor* b);
void add_cpu_forward(Tensor* a, Tensor* b, Tensor* out);
void add_gpu_forward(Tensor* a, Tensor* b, Tensor* out);

Tensor* tensor_mul(Tensor* a, Tensor* b);
void mul_cpu_forward(Tensor* a, Tensor* b, Tensor* out);
void mul_gpu_forward(Tensor* a, Tensor* b, Tensor* out);

Tensor* tensor_add_bias(Tensor* a, Tensor* bias);
void bias_cpu_forward(Tensor* a, Tensor* bias, Tensor* out);
void bias_gpu_forward(Tensor* a, Tensor* bias, Tensor* out);

Tensor* tensor_matmul(Tensor* a, Tensor* b);
void matmul_cpu_forward(Tensor* a, Tensor* b, Tensor* out);
void matmul_gpu_forward(Tensor* a, Tensor* b, Tensor* out);

Tensor* tensor_relu(Tensor* a);
void relu_cpu_forward(Tensor* a, Tensor* out);
void relu_gpu_forward(Tensor* a, Tensor* out);

#endif