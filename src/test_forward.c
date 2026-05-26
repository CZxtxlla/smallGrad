#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../include/tensor.h"
#include "../include/ops.h"

static int equal_float_arr(const float* a, const float* b, size_t n, float tol) {
    for (size_t i = 0; i < n; i++) {
        if (fabsf(a[i] - b[i]) > tol) return 0;
    }
    return 1;
}

static void fill_tensor(Tensor* t, const float* values) {
    for (size_t i = 0; i < t->size; i++) {
        t->cpu_data[i] = values[i];
    }
}

static Tensor* make_cpu_tensor(int* shape, int ndims, const float* values, bool requires_grad) {
    Tensor* t = create_tensor(shape, ndims, DEVICE_CPU, requires_grad);
    if (!t) return NULL;
    fill_tensor(t, values);
    return t;
}

static Tensor* make_gpu_tensor(int* shape, int ndims, const float* values, bool requires_grad) {
    Tensor* t = make_cpu_tensor(shape, ndims, values, requires_grad);
    if (!t) return NULL;
    tensor_to_device(t, DEVICE_GPU);
    return t;
}

static void check_cpu_tensor(const char* label, Tensor* t, const float* expected, size_t n) {
    if (!t || !t->cpu_data || !equal_float_arr(t->cpu_data, expected, n, 1e-5f)) {
        fprintf(stderr, "%s mismatch\n", label);
        exit(1);
    }
}

static void check_gpu_output(const char* label, Tensor* t, const float* expected, size_t n) {
    tensor_to_device(t, DEVICE_CPU);
    check_cpu_tensor(label, t, expected, n);
}

int main() {
    printf("--- Testing Forward Pass Dispatcher ---\n\n");

    // CPU reference cases
    int shape2[2] = {2, 2};
    float a_vals[4] = {1, 2, 3, 4};
    float b_vals[4] = {2, 0, 1, 2};
    Tensor* a_cpu = make_cpu_tensor(shape2, 2, a_vals, false);
    Tensor* b_cpu = make_cpu_tensor(shape2, 2, b_vals, false);
    if (!a_cpu || !b_cpu) return 1;

    Tensor* add_cpu = tensor_add(a_cpu, b_cpu);
    Tensor* mul_cpu = tensor_mul(a_cpu, b_cpu);
    float relu_input[4] = {-1, 2, 3, 4};
    Tensor* relu_cpu_in = make_cpu_tensor(shape2, 2, relu_input, false);
    Tensor* relu_cpu = tensor_relu(relu_cpu_in);

    int a3_shape[2] = {2, 3};
    int b3_shape[2] = {3, 2};
    float A_vals[6] = {1, 2, 3, 4, 5, 6};
    float B_vals[6] = {7, 8, 9, 10, 11, 12};
    Tensor* A_cpu = make_cpu_tensor(a3_shape, 2, A_vals, false);
    Tensor* B_cpu = make_cpu_tensor(b3_shape, 2, B_vals, false);
    Tensor* mat_cpu = tensor_matmul(A_cpu, B_cpu);

    int bias_shape[1] = {3};
    int biased_shape[2] = {2, 3};
    float X_vals[6] = {1, 2, 3, 4, 5, 6};
    float bias_vals[3] = {10, 20, 30};
    Tensor* X_cpu = make_cpu_tensor(biased_shape, 2, X_vals, false);
    Tensor* bias_cpu = make_cpu_tensor(bias_shape, 1, bias_vals, false);
    Tensor* add_bias_cpu = tensor_add_bias(X_cpu, bias_cpu);

    check_cpu_tensor("tensor_add CPU", add_cpu, (float[]){3, 2, 4, 6}, 4);
    check_cpu_tensor("tensor_mul CPU", mul_cpu, (float[]){2, 0, 3, 8}, 4);
    check_cpu_tensor("tensor_relu CPU", relu_cpu, (float[]){0, 2, 3, 4}, 4);
    check_cpu_tensor("tensor_matmul CPU", mat_cpu, (float[]){58, 64, 139, 154}, 4);
    check_cpu_tensor("tensor_add_bias CPU", add_bias_cpu, (float[]){11, 22, 33, 14, 25, 36}, 6);

    printf("[CPU] forward ops: PASS\n");

    // GPU forward numeric checks via tensor_to_device
    Tensor* a_gpu = make_gpu_tensor(shape2, 2, a_vals, false);
    Tensor* b_gpu = make_gpu_tensor(shape2, 2, b_vals, false);
    Tensor* add_gpu = tensor_add(a_gpu, b_gpu);
    check_gpu_output("tensor_add GPU", add_gpu, (float[]){3, 2, 4, 6}, 4);

    a_gpu = make_gpu_tensor(shape2, 2, a_vals, false);
    b_gpu = make_gpu_tensor(shape2, 2, b_vals, false);
    Tensor* mul_gpu = tensor_mul(a_gpu, b_gpu);
    check_gpu_output("tensor_mul GPU", mul_gpu, (float[]){2, 0, 3, 8}, 4);

    float relu_gpu_input[4] = {-1, 2, -3, 4};
    Tensor* relu_gpu_in = make_gpu_tensor(shape2, 2, relu_gpu_input, false);
    Tensor* relu_gpu = tensor_relu(relu_gpu_in);
    check_gpu_output("tensor_relu GPU", relu_gpu, (float[]){0, 2, 0, 4}, 4);

    a_gpu = make_gpu_tensor(a3_shape, 2, A_vals, false);
    b_gpu = make_gpu_tensor(b3_shape, 2, B_vals, false);
    Tensor* mat_gpu = tensor_matmul(a_gpu, b_gpu);
    check_gpu_output("tensor_matmul GPU", mat_gpu, (float[]){58, 64, 139, 154}, 4);

    float X_gpu_vals[6] = {1, 2, 3, 4, 5, 6};
    float bias_gpu_vals[3] = {10, 20, 30};
    Tensor* X_gpu = make_gpu_tensor(biased_shape, 2, X_gpu_vals, false);
    Tensor* bias_gpu = make_gpu_tensor(bias_shape, 1, bias_gpu_vals, false);
    Tensor* add_bias_gpu = tensor_add_bias(X_gpu, bias_gpu);
    check_gpu_output("tensor_add_bias GPU", add_bias_gpu, (float[]){11, 22, 33, 14, 25, 36}, 6);

    printf("[GPU] forward ops: PASS\n");

    free_tensor(a_cpu);
    free_tensor(b_cpu);
    free_tensor(add_cpu);
    free_tensor(mul_cpu);
    free_tensor(relu_cpu_in);
    free_tensor(relu_cpu);
    free_tensor(A_cpu);
    free_tensor(B_cpu);
    free_tensor(mat_cpu);
    free_tensor(X_cpu);
    free_tensor(bias_cpu);
    free_tensor(add_bias_cpu);
    free_tensor(a_gpu);
    free_tensor(b_gpu);
    free_tensor(add_gpu);
    free_tensor(mul_gpu);
    free_tensor(relu_gpu_in);
    free_tensor(relu_gpu);
    free_tensor(X_gpu);
    free_tensor(bias_gpu);
    free_tensor(add_bias_gpu);
    free_tensor(mat_gpu);

    printf("Forward Pass Tests Completed Successfully!\n");
    return 0;
}