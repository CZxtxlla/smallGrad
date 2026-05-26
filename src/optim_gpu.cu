#include "../include/optim.h"
#include <cuda_runtime.h>
#include "../include/cuda_utils.h"


// -------- kernels ------------

__global__ void sgd_step_kernel(float* data, float* grad, float lr, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size) {
        data[i] -= lr * grad[i];
    }
}


// ---------- launchers ------------

void sgd_gpu_step(Tensor* parameter, float lr) {
    if (parameter->gpu_grad != NULL) {
        int threads = 256;
        dim3 dimBlock(threads, 1, 1);
        dim3 dimGrid((parameter->size + threads - 1)/threads, 1, 1);

        sgd_step_kernel<<<dimGrid, dimBlock>>>(parameter->gpu_data, parameter->gpu_grad, lr, parameter->size);
        CUDA_CHECK_GOTO(cudaGetLastError(), cleanup);
    }
    return;

cleanup:
    exit(EXIT_FAILURE);
}

void sgd_gpu_zero_grad(Tensor* parameter) {
    if (parameter->gpu_grad != NULL) {
        size_t bytes = parameter->size * sizeof(float);
        CUDA_CHECK_GOTO(cudaMemset(parameter->gpu_grad, 0, bytes), cleanup);
    }
    return;

cleanup:
    exit(EXIT_FAILURE);
}