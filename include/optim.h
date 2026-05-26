#ifndef OPTIM_H
#define OPTIM_H

#include "tensor.h"
#include "ops.h"

#ifdef __cplusplus
extern "C" {
#endif

// --------- Stochastic gradient descent optimizer ---------

typedef struct {
    Tensor** parameters; // pointer to array of tensors we want to update (weights, biases)
    int num_parameters; // number of tensors in the array
    float lr; // learning rate
} SGD;

SGD* sgd_create(Tensor** parameters, int num_parameters, float lr); // creates optimizer

void sgd_step(SGD* optim); // applies learning rule to all parameters, (data = data - (lr * grad))
void sgd_gpu_step(Tensor* parameter, float lr);
void sgd_cpu_step(Tensor* parameter, float lr);


void sgd_zero_grad(SGD* optim); // resets all gradients to 0 before next forward pass
void sgd_cpu_zero_grad(Tensor* parameter);
void sgd_gpu_zero_grad(Tensor* parameter);

void sgd_free(SGD* optim); // free optimizer, but not tensors it points to

// --------- other potential optimizers -----------




#ifdef __cplusplus
}
#endif


#endif