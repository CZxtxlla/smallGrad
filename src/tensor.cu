#include "../include/tensor.h"
#include <cuda_runtime.h>
#include "../include/cuda_utils.h"




Tensor* create_tensor(int* shape, int ndims, DeviceType device, bool requires_grad) {
    // Allocate and initialize one tensor given the input parameters

    Tensor* t = (Tensor*) calloc(1, sizeof(Tensor)); // creates memory for 1 tensor device, all intialized to 0
    if (t == NULL) {
        fprintf(stderr, "Error: failed to allocate memory for tensor.\n");
        return NULL;
    }
    t->ndims = ndims;
    t->requires_grad = requires_grad;
    t->op = OP_NONE;
    t->parents = NULL;
    t->num_parents = 0;

    t->size = 1; // initialize to accumulate dimensions via multipication
    t->shape = (int*)malloc(ndims * sizeof(int));
    if (t->shape == NULL) {
        fprintf(stderr, "Error: failed to allocate memory for shape.\n");
        goto cleanup;
    }

    for (int i = 0; i < ndims; i++) {
        t->shape[i] = shape[i];
        t->size *= shape[i];
    }

    t->device = device;
    if (device == DEVICE_CPU) {
        t->gpu_data = NULL;
        t->gpu_grad = NULL;
        t->cpu_data = (float*) calloc(t->size, sizeof(float)); //everything set to 0
        if (t->cpu_data == NULL) {
            fprintf(stderr, "Error: failed to allocate memory for cpu data.\n");
            goto cleanup;
        }

        if (requires_grad) {
            t->cpu_grad = (float*) calloc(t->size, sizeof(float));
            if (t->cpu_grad == NULL) {
                fprintf(stderr, "Error: failed to allocate memory for cpu tensor grad.\n");
                goto cleanup;
            }
        } else {
            t->cpu_grad = NULL;
        }

    } else if (device == DEVICE_GPU) {
        t->cpu_data = NULL;
        t->cpu_grad = NULL;


        CUDA_CHECK_GOTO(cudaMalloc((void**) &t->gpu_data, t->size * sizeof(float)), cleanup); 
        CUDA_CHECK_GOTO(cudaMemset(t->gpu_data, 0, t->size * sizeof(float)), cleanup);

        
        if (requires_grad) {
            CUDA_CHECK_GOTO(cudaMalloc((void**) &t->gpu_grad, t->size * sizeof(float)), cleanup);
            CUDA_CHECK_GOTO(cudaMemset(t->gpu_grad, 0, t->size * sizeof(float)), cleanup);
        } else {
            t->gpu_grad = NULL;
        }
    } else {
        fprintf(stderr, "Error: Not a valid device.\n");
        goto cleanup;
    }

    return t;

cleanup: 
    free_tensor(t);
    return NULL;
}

void free_tensor(Tensor* t) {
    // Free all allocated memory corresponding to a tensor
    
    if (t == NULL) {
        return;
    }

    // Free cpu data/grad
    free(t->cpu_data);
    free(t->cpu_grad);
    // Free gpu data/grad
    cudaFree(t->gpu_data);
    cudaFree(t->gpu_grad);
    // Free rest
    free(t->shape);
    free(t->parents);
    // Finally, free tensor
    free(t);
}
