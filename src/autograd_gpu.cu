#include "../include/autograd.h"
#include <cuda_runtime.h>
#include "../include/cuda_utils.h"


// ------------- Kernels ----------------


__global__ void backward_add_kernel(float* t_grad, float* a_grad, float* b_grad, int size) {
    // takes both grad arrays for the tensor and a parent and passes on the gradient according to addition
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size) {
        if (a_grad != NULL) {
            atomicAdd(&a_grad[i], t_grad[i]);
        }
        if (b_grad != NULL) {
            atomicAdd(&b_grad[i], t_grad[i]);
        } 
    }
}







// -------------- Helpers ---------------

void backward_gpu_add(Tensor* t, Tensor* a, Tensor* b) {
    // calls the kernel to pass the gradient backwards on the gpu
    int threads = 256;
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid((a->size + threads - 1)/threads, 1, 1);
    
    backward_add_kernel<<<dimGrid, dimBlock>>>(t->gpu_grad, a->gpu_grad, b->gpu_grad, t->size);
    CUDA_CHECK_GOTO(cudaGetLastError(), cleanup);
    
    return;

cleanup:
    exit(EXIT_FAILURE);
}