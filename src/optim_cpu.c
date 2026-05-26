#include "../include/optim.h"

void sgd_cpu_step(Tensor* parameter, float lr) {
    if (parameter->cpu_grad != NULL) {
        for (int i = 0; i < parameter->size; i++) {
            parameter->cpu_data[i] -= lr * parameter->cpu_grad[i];
        }
    }
}

void sgd_cpu_zero_grad(Tensor* parameter) {
    size_t bytes = parameter->size * sizeof(float);\
    memset(parameter->cpu_grad, 0, bytes);
}