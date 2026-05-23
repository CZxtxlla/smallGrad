#ifndef AUTOGRAD_H
#define AUTOGRAD_H

#include "ops.h"

void backward_add(Tensor* t);
void backward_cpu_add(Tensor* t, Tensor* a, Tensor* b);
void backward_gpu_add(Tensor* t, Tensor* a, Tensor* b);

void backward_mul(Tensor* t);
void backward_cpu_mul(Tensor* t, Tensor* a, Tensor* b);
void backward_gpu_mul(Tensor* t, Tensor* a, Tensor* b);

void backward_add_bias(Tensor* t);
void backward_cpu_add_bias(Tensor* t, Tensor* a, Tensor* bias);
void backward_gpu_add_bias(Tensor* t, Tensor* a, Tensor* bias);

void backward_matmul(Tensor* t);
void backward_cpu_matmul(Tensor* t, Tensor* a, Tensor* b);
void backward_gpu_matmul(Tensor* t, Tensor* a, Tensor* b);

void backward_relu(Tensor* t);
void backward_cpu_relu(Tensor* t, Tensor* a);
void backward_gpu_relu(Tensor* t, Tensor* a);



void backward(Tensor* t);


#endif