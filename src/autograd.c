#include "../include/autograd.h"

void backward_add(Tensor* t) {
    // Takes tensor t which is the result of an addition operation and computes gradients for its parents

    if (t->op != OP_ADD || t->num_parents != 2) {
        fprtinf(stderr, "Error: backward_add called on tensor that is not the result of an addition operation.\n");
        return NULL;
    }

    Tensor* a = t->parents[0];
    Tensor* b = t->parents[1];

    if (t->device == DEVICE_CPU) {
        backward_cpu_add(t, a, b);
    } else if (t->device == DEVICE_GPU) {
        backward_gpu_add(t, a, b);
    }
}

void backward_mul(Tensor* t) {
    // Takes tensor t which is the result of a multiplication operation and computes gradients for its parents

    if (t->op != OP_MUL || t->num_parents != 2) {
        fprtinf(stderr, "Error: backward_add called on tensor that is not the result of a multiplication operation.\n");
        return NULL;
    }

    Tensor* a = t->parents[0];
    Tensor* b = t->parents[1];

    if (t->device == DEVICE_CPU) {
        backward_cpu_mul(t, a, b);
    } else if (t->device == DEVICE_GPU) {
        backward_gpu_mul(t, a, b);
    }
}

void backward_add_bias(Tensor* t) {
    // Takes tensor t which is the result of a multiplication operation and computes gradients for its parents

    if (t->op != OP_ADDBIAS || t->num_parents != 2) {
        fprtinf(stderr, "Error: backward_add called on tensor that is not the result of an add bias operation.\n");
        return NULL;
    }

    Tensor* a = t->parents[0];
    Tensor* bias = t->parents[1];

    if (t->device == DEVICE_CPU) {
        backward_cpu_add_bias(t, a, bias);
    } else if (t->device == DEVICE_GPU) {
        backward_gpu_add_bias(t, a, bias);
    }
}

void backward_matmul(Tensor* t) {
    // Takes tensor t which is the result of a matrix multiplication operation and computes gradients for its parents

    if (t->op != OP_MATMUL || t->num_parents != 2) {
        fprtinf(stderr, "Error: backward_add called on tensor that is not the result of a matrix multiplication operation.\n");
        return NULL;
    }

    Tensor* a = t->parents[0];
    Tensor* b = t->parents[1];

    if (t->device == DEVICE_CPU) {
        backward_cpu_matmul(t, a, b);
    } else if (t->device == DEVICE_GPU) {
        backward_gpu_matmul(t, a, b);
    }
}

void backward_relu(Tensor* t) {
    // Takes tensor t which is the result of a multiplication operation and computes gradients for its parents

    if (t->op != OP_RELU || t->num_parents != 1) {
        fprtinf(stderr, "Error: backward_add called on tensor that is not the result of a relu operation.\n");
        return NULL;
    }

    Tensor* a = t->parents[0];

    if (t->device == DEVICE_CPU) {
        backward_cpu_relu(t, a);
    } else if (t->device == DEVICE_GPU) {
        backward_gpu_relu(t, a);
    }
}

void backward(Tensor* t) {
    
}
