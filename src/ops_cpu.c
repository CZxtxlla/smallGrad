#include "../include/ops.h"

void add_cpu_forward(Tensor* a, Tensor* b, Tensor* out) {
    for (int i = 0; i < a->size; i++) {
        out->cpu_data[i] = a->cpu_data[i] + b->cpu_data[i];
    }
}

void mul_cpu_forward(Tensor* a, Tensor* b, Tensor* out) {
    for (int i = 0; i < a->size; i++) {
        out->cpu_data[i] = a->cpu_data[i] * b->cpu_data[i];
    }
}

void bias_cpu_forward(Tensor* a, Tensor* bias, Tensor* out) {
    for (int i = 0; i < a->shape[0]; i++) {
        for (int j = 0; j < a->shape[1]; j++) {
            out->cpu_data[i * a->shape[1] + j] = a->cpu_data[i * a->shape[1] + j] + bias->cpu_data[j];
        }
    }
}

void matmul_cpu_forward(Tensor* a, Tensor* b, Tensor* out) {
    for (int i = 0; i < a->shape[0]; i++) {
        for (int j = 0; j < b->shape[1]; j++) {
            float sum = 0.0f;
            for (int k = 0; k < a -> shape[1]; k++) {
                sum += a->cpu_data[i * a->shape[1] + k]* b->cpu_data[k * b->shape[1] + j];
            }
            out->cpu_data[i * b->shape[1] + j] = sum;
        }
    }
}

void relu_cpu_forward(Tensor* a, Tensor* out) {
    for (int i = 0; i < a->size; i++) {
        out->cpu_data[i] = a->cpu_data[i] > 0 ? a->cpu_data[i] : 0.0f;
    }
}

void mse_cpu_forward(Tensor* pred, Tensor* target, Tensor* out) {
    float sum = 0.0f;
    for (int i = 0; i < pred->size; i++) {
        float diff = pred->cpu_data[i] - target->cpu_data[i];
        sum += diff * diff;
    }
    out->cpu_data[0] = sum/pred->size;
}