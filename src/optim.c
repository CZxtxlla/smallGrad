#include "../include/optim.h"


SGD* sgd_create(Tensor** parameters, int num_parameters, float lr) {
    // Allocates the optimizer and stores the pointers to the learnable parameters
    SGD* optim = (SGD*)malloc(sizeof(SGD));
    if (optim == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for SGD optimizer.\n");
        return NULL;
    }
    optim->parameters = parameters;
    optim->num_parameters = num_parameters;
    optim->lr = lr;

    return optim;
}

void sgd_step(SGD* optim) {
    for (int i = 0; i < optim->num_parameters; i++) {
        Tensor* param = optim->parameters[i];
        if (param->requires_grad && param->device == DEVICE_CPU) {
            sgd_cpu_step(param, optim->lr);
        } else if (param->requires_grad && param->device == DEVICE_GPU) {
            sgd_gpu_step(param, optim->lr);
        }
    }
}

void sgd_zero_grad(SGD* optim) {
    if (optim == NULL) {
        return;
    }

    for (int i = 0; i < optim->num_parameters; i++) {
        Tensor* param = optim->parameters[i];

        if (param->requires_grad && param->device == DEVICE_CPU) {
            sgd_cpu_zero_grad(param);
        } else if (param->requires_grad && param->device == DEVICE_GPU) {
            sgd_gpu_zero_grad(param);
        }
    }
}

void sgd_free(SGD* optim) {
    free(optim);
}