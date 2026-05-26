#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../include/tensor.h"
#include "../include/ops.h"
#include "../include/autograd.h"

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

static void move_to_cpu_and_check_grad(Tensor* t, const float* expected, const char* label) {
    tensor_to_device(t, DEVICE_CPU);
    if (!t->cpu_grad || !equal_float_arr(t->cpu_grad, expected, t->size, 1e-4f)) {
        fprintf(stderr, "%s gradient mismatch\nexpected:", label);
        for (size_t i = 0; i < t->size; i++) fprintf(stderr, " %g", expected[i]);
        fprintf(stderr, "\nactual:");
        for (size_t i = 0; i < t->size; i++) fprintf(stderr, " %g", t->cpu_grad ? t->cpu_grad[i] : -9999.0f);
        fprintf(stderr, "\n");
        exit(1);
    }
}

static void seed_gpu_grad_from_host(Tensor* t, const float* grad_values) {
    tensor_to_device(t, DEVICE_CPU);
    for (size_t i = 0; i < t->size; i++) {
        t->cpu_grad[i] = grad_values[i];
    }
    tensor_to_device(t, DEVICE_GPU);
}

static void check_cpu_forward(const char* label, Tensor* t, const float* expected) {
    if (!t || !t->cpu_data || !equal_float_arr(t->cpu_data, expected, t->size, 1e-5f)) {
        fprintf(stderr, "%s forward mismatch\n", label);
        exit(1);
    }
}

int main() {
    printf("--- Testing Backward Pass & Autograd ---\n\n");

    int shape2[2] = {2, 2};
    float x_vals[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float y_vals[4] = {5.0f, 6.0f, 7.0f, 8.0f};

    // CPU reference: z = (x + y) * x
    Tensor* x_cpu = make_cpu_tensor(shape2, 2, x_vals, true);
    Tensor* y_cpu = make_cpu_tensor(shape2, 2, y_vals, true);
    Tensor* sum_cpu = tensor_add(x_cpu, y_cpu);
    Tensor* z_cpu = tensor_mul(sum_cpu, x_cpu);

    check_cpu_forward("CPU sum", sum_cpu, (float[]){6, 8, 10, 12});
    check_cpu_forward("CPU z", z_cpu, (float[]){6, 16, 30, 48});

    for (size_t i = 0; i < z_cpu->size; i++) z_cpu->cpu_grad[i] = 1.0f;
    backward(z_cpu);

    float expected_dx[4];
    float expected_dy[4];
    for (size_t i = 0; i < 4; i++) {
        expected_dx[i] = 2.0f * x_vals[i] + y_vals[i];
        expected_dy[i] = x_vals[i];
    }
    if (!equal_float_arr(x_cpu->cpu_grad, expected_dx, 4, 1e-4f) || !equal_float_arr(y_cpu->cpu_grad, expected_dy, 4, 1e-4f)) {
        fprintf(stderr, "CPU backward computation mismatch\n");
        return 1;
    }
    printf("[CPU] backward z=(x+y)*x: PASS\n");

    // CPU reference: self-mul accumulation test z = x * x
    Tensor* x2_cpu = make_cpu_tensor(shape2, 2, x_vals, true);
    Tensor* sq_cpu = tensor_mul(x2_cpu, x2_cpu);
    for (size_t i = 0; i < sq_cpu->size; i++) sq_cpu->cpu_grad[i] = 1.0f;
    backward(sq_cpu);
    float expected_sq_dx[4] = {2, 4, 6, 8};
    if (!equal_float_arr(x2_cpu->cpu_grad, expected_sq_dx, 4, 1e-4f)) {
        fprintf(stderr, "CPU self-mul backward mismatch\n");
        return 1;
    }
    printf("[CPU] backward z=x*x: PASS\n");

    // GPU: addition backward
    Tensor* gx = make_gpu_tensor(shape2, 2, x_vals, true);
    Tensor* gy = make_gpu_tensor(shape2, 2, y_vals, true);
    Tensor* gsum = tensor_add(gx, gy);
    tensor_to_device(gsum, DEVICE_CPU);
    check_cpu_forward("GPU add forward", gsum, (float[]){6, 8, 10, 12});
    seed_gpu_grad_from_host(gsum, (float[]){1, 1, 1, 1});
    backward(gsum);
    float expected_add_dx[4] = {1, 1, 1, 1};
    float expected_add_dy[4] = {1, 1, 1, 1};
    move_to_cpu_and_check_grad(gx, expected_add_dx, "GPU add x");
    move_to_cpu_and_check_grad(gy, expected_add_dy, "GPU add y");
    printf("[GPU] backward add: PASS\n");

    // GPU: multiply backward with non-uniform upstream gradient
    gx = make_gpu_tensor(shape2, 2, x_vals, true);
    gy = make_gpu_tensor(shape2, 2, y_vals, true);
    Tensor* gmul = tensor_mul(gx, gy);
    tensor_to_device(gmul, DEVICE_CPU);
    check_cpu_forward("GPU mul forward", gmul, (float[]){5, 12, 21, 32});
    seed_gpu_grad_from_host(gmul, (float[]){1, 2, 3, 4});
    backward(gmul);
    float expected_mul_dx[4] = {5, 12, 21, 32};
    float expected_mul_dy[4] = {1, 4, 9, 16};
    move_to_cpu_and_check_grad(gx, expected_mul_dx, "GPU mul x");
    move_to_cpu_and_check_grad(gy, expected_mul_dy, "GPU mul y");
    printf("[GPU] backward mul: PASS\n");

    // GPU: relu backward
    float relu_vals[4] = {-3.0f, -1.0f, 2.0f, 4.0f};
    Tensor* grelu = make_gpu_tensor(shape2, 2, relu_vals, true);
    Tensor* grelu_out = tensor_relu(grelu);
    tensor_to_device(grelu_out, DEVICE_CPU);
    check_cpu_forward("GPU relu forward", grelu_out, (float[]){0, 0, 2, 4});
    seed_gpu_grad_from_host(grelu_out, (float[]){1, 2, 3, 4});
    backward(grelu_out);
    float expected_relu_dx[4] = {0, 0, 3, 4};
    move_to_cpu_and_check_grad(grelu, expected_relu_dx, "GPU relu x");
    printf("[GPU] backward relu: PASS\n");

    // GPU: matrix multiplication backward
    int a_shape[2] = {2, 3};
    int b_shape[2] = {3, 2};
    float A_vals[6] = {1, 2, 3, 4, 5, 6};
    float B_vals[6] = {7, 8, 9, 10, 11, 12};
    Tensor* gA = make_gpu_tensor(a_shape, 2, A_vals, true);
    Tensor* gB = make_gpu_tensor(b_shape, 2, B_vals, true);
    Tensor* gmat = tensor_matmul(gA, gB);
    tensor_to_device(gmat, DEVICE_CPU);
    check_cpu_forward("GPU matmul forward", gmat, (float[]){58, 64, 139, 154});
    seed_gpu_grad_from_host(gmat, (float[]){1, 2, 3, 4});
    backward(gmat);
    float expected_gA_grad[6] = {
        1*7 + 2*8, 1*9 + 2*10, 1*11 + 2*12,
        3*7 + 4*8, 3*9 + 4*10, 3*11 + 4*12
    };
    float expected_gB_grad[6] = {
        13, 18,
        17, 24,
        21, 30
    };
    move_to_cpu_and_check_grad(gA, expected_gA_grad, "GPU matmul A");
    move_to_cpu_and_check_grad(gB, expected_gB_grad, "GPU matmul B");
    printf("[GPU] backward matmul: PASS\n");

    // GPU: bias addition backward
    int bias_shape[1] = {3};
    int batch_shape[2] = {2, 3};
    float X_vals[6] = {1, 2, 3, 4, 5, 6};
    float bias_vals[3] = {10, 20, 30};
    Tensor* gx2 = make_gpu_tensor(batch_shape, 2, X_vals, true);
    Tensor* gbias = make_gpu_tensor(bias_shape, 1, bias_vals, true);
    Tensor* gbias_out = tensor_add_bias(gx2, gbias);
    tensor_to_device(gbias_out, DEVICE_CPU);
    check_cpu_forward("GPU add_bias forward", gbias_out, (float[]){11, 22, 33, 14, 25, 36});
    seed_gpu_grad_from_host(gbias_out, (float[]){1, 2, 3, 4, 5, 6});
    backward(gbias_out);
    float expected_x2_grad[6] = {1, 2, 3, 4, 5, 6};
    float expected_bias_grad[3] = {1+4, 2+5, 3+6};
    move_to_cpu_and_check_grad(gx2, expected_x2_grad, "GPU add_bias x");
    move_to_cpu_and_check_grad(gbias, expected_bias_grad, "GPU add_bias bias");
    printf("[GPU] backward add_bias: PASS\n");

    free_tensor(x_cpu);
    free_tensor(y_cpu);
    free_tensor(sum_cpu);
    free_tensor(z_cpu);
    free_tensor(x2_cpu);
    free_tensor(sq_cpu);
    free_tensor(gx);
    free_tensor(gy);
    free_tensor(gsum);
    free_tensor(gmul);
    free_tensor(grelu);
    free_tensor(grelu_out);
    free_tensor(gA);
    free_tensor(gB);
    free_tensor(gmat);
    free_tensor(gx2);
    free_tensor(gbias);
    free_tensor(gbias_out);

    printf("Backward Pass Tests Completed Successfully!\n");
    return 0;
}