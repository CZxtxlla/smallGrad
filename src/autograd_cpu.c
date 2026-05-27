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
    int batch_size = a->shape[0]; // rows
    int features = a->shape[1]; // columns

    if (a->requires_grad) {
        for (int i = 0; i < a->size; i++) {
            a->cpu_grad[i] += t->cpu_grad[i];
        }
    }
    if (bias->requires_grad) {
        for (int i = 0; i < batch_size; i++) {
            for (int j = 0; j < features; j++) {
                bias->cpu_grad[j] += t->cpu_grad[i * features + j];
            }
        }
    }

}

void backward_cpu_matmul(Tensor* t, Tensor* a, Tensor* b) {
    // performs the backwards transmission of gradient on cpu for matrix multiplication
    int j = a->shape[0];
    int k = a->shape[1];
    int l = b->shape[1];

    if (a->requires_grad) {
        for (int m = 0; m < j; m++) {
            for (int n = 0; n < k; n++) {
                float grad_a_mn = 0.0f;
                for (int p = 0; p < l; p++) {
                    grad_a_mn += t->cpu_grad[m * l + p] * b->cpu_data[n * l + p]; // dot product of mth row of gradT and nth row of B (b/c transpose)
                }
                a->cpu_grad[m * k + n] += grad_a_mn;
            }
        }
    }

    if (b->requires_grad) {
        for (int m = 0; m < k; m++) {
            for (int n = 0; n < l; n++) {
                float grad_b_mn= 0.0f;
                for (int p = 0; p < j; p++) {
                    grad_b_mn += a->cpu_data[p * k + m] * t->cpu_grad[p * l + n];
                }
                b->cpu_grad[m * l + n] += grad_b_mn;
            }
        }
    }
}

void backward_cpu_relu(Tensor* t, Tensor* a) {
    // performs the backwards transmission of gradient on cpu for relu
    if (a->requires_grad) {
        for (int i = 0; i < t->size; i++) {
            a->cpu_grad[i] += a->cpu_data[i] > 0 ? t ->cpu_grad[i] : 0.0f; // grad input = grad output if input > 0
        }
    }
}

void backward_cpu_mse(Tensor* t, Tensor* pred, Tensor* target) {
    // performs backwards propogation of gradient on cpu for mse
    float scale = 2.0f / pred->size;

    if (pred->requires_grad) {
        for (int i = 0; i < pred->size; i++) {
            pred->cpu_grad[i] += scale * (pred->cpu_data[i] - target->cpu_data[i]) * t->cpu_grad[0];
        }
    } 
    if (target->requires_grad) {
        for (int i = 0; i < target->size; i++) {
            target->cpu_grad[i] += -scale * (pred->cpu_data[i] - target->cpu_data[i]) * t->cpu_grad[0];
        }
    }

}

void backward_cpu_cross_entropy(Tensor* t, Tensor* pred, Tensor* target) {
    int batch_size = pred->shape[0];
    float scale = t->cpu_grad[0] / batch_size;

    if (pred->requires_grad) {
        for (int i = 0; i < pred->size; i++) {
            // combined derivative
            pred->cpu_grad[i] += (pred->cpu_data[i] - target->cpu_data[i]) * scale;
        }
    }
}