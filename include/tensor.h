#ifndef TENSOR_H
#define TENSOR_H

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

typedef enum {
    DEVICE_CPU,
    DEVICE_GPU
} DeviceType;

typedef enum {
    OP_NONE,
    OP_ADD,
    OP_MUL,
    OP_MATMUL,
    OP_RELU,
    OP_ADDBIAS
} opType;

typedef struct Tensor {
    // CPU memory pointers
    float* cpu_data;
    float* cpu_grad;

    // GPU memory pointers 
    float* gpu_data;
    float* gpu_grad;

    DeviceType device; // either DEVICE_CPU or DEVICE_GPU

    int* shape; // shape of the tensor, i.e. [64, 128] for 2D matrix with 64 elements in column, 128 in row
    int ndims; // Number of dimensions
    size_t size; // total number of elements

    // Autograd graph data
    bool requires_grad; 
    struct Tensor** parents; // list of pointers to parent tensors
    int num_parents; // greater than or equal to 1
    opType op; // type of operation used to create this tensor from the parent tensor(s)
} Tensor;


Tensor* create_tensor(int* shape, int ndims, DeviceType device, bool requires_grad);
void free_tensor(Tensor* t);

void tensor_to_device(Tensor* t, DeviceType device); // sends tensor to be stored on specified device

#endif