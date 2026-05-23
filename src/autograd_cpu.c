#include "../include/autograd.h"


void backward_cpu_add(Tensor* t, Tensor* a, Tensor* b) {
    // performs the backwards transmission of gradient on cpu for addition
    if (a->requires_grad) {
        for (int i = 0; i < a->size; i++) {
            a->cpu_grad[i] += t->cpu_grad[i]; 
        }
    }
    if (b->requires_grad) {
        for (int i = 0; i < b->size; i++) {
            b->cpu_grad[i] += t->cpu_grad[i]; 
        }
    }
}

void backward_cpu_mul(Tensor* t, Tensor* a, Tensor* b) {
    // performs the backwards transmission of gradient on cpu for multiplication
    if (a->requires_grad) {
        for (int i = 0; i < a -> size; i++) {
            a->cpu_grad[i] += b->cpu_data[i] * t->cpu_grad[i];
        }
    }
    if (b->requires_grad) {
        for (int i = 0; i < a -> size; i++) {
            b->cpu_grad[i] += a->cpu_data[i] * t->cpu_grad[i];
        }
    }
}

void backward_cpu_add_bias(Tensor* t, Tensor* a, Tensor* bias) {
    // performs the backwards transmission of gradient on cpu for bias addition
}

void backward_cpu_matmul(Tensor* t, Tensor* a, Tensor* b) {
    // performs the backwards transmission of gradient on cpu for matrix multiplication

}

void backward_cpu_relu(Tensor* t, Tensor* a) {
    // performs the backwards transmission of gradient on cpu for relu

}