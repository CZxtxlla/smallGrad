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

void cross_entropy_cpu_forward(Tensor* pred, Tensor* target, Tensor* out) {
    int batch_size = pred->shape[0];
    int num_classes = pred->shape[1];

    float total_loss = 0.0f;

    for (int b = 0; b < batch_size; b++) {
        int offset = b * num_classes;

        // compute max
        float max_val = pred->cpu_data[offset];
        for (int c = 1; c < num_classes; c++) {
            if (pred->cpu_data[offset + c] > max_val) {
                max_val = pred->cpu_data[offset + c];
            }
        }
        // compute exponentials
        float exp_sum = 0.0f;
        for (int c = 0; c < num_classes; c++) {
            float e = expf(pred->cpu_data[offset + c] - max_val);
            pred->cpu_data[offset + c] = e;
            exp_sum +=e;
        }

        // normalize to probabilities and compute loss, -log(p) of the probability of the true class
        for (int c = 0; c < num_classes; c++) {
            float prob = pred->cpu_data[offset + c] / exp_sum;
            pred->cpu_data[offset + c] = prob;

            if (target->cpu_data[offset + c] == 1.0f) {
                total_loss -= logf(prob + 1e-7f);
            }
        }
    }
    out->cpu_data[0] = total_loss/batch_size;
}