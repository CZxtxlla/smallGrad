#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../include/tensor.h"
#include "../include/ops.h"
#include "../include/autograd.h"
#include "../include/nn.h"
#include "../include/optim.h"

static Tensor* make_tensor(int* shape, int ndims, const float* values, bool requires_grad, DeviceType device) {
    Tensor* t = create_tensor(shape, ndims, DEVICE_CPU, requires_grad);
    if (t == NULL) {
        return NULL;
    }

    for (size_t i = 0; i < t->size; i++) {
        t->cpu_data[i] = values[i];
    }

    if (device == DEVICE_GPU) {
        tensor_to_device(t, DEVICE_GPU);
    }

    return t;
}

static float tensor_scalar_value(Tensor* t) {
    if (t->device == DEVICE_GPU) {
        tensor_to_device(t, DEVICE_CPU);
    }

    return t->cpu_data[0];
}

static void seed_loss_grad(Tensor* loss) {
    if (loss->device == DEVICE_CPU) {
        loss->cpu_grad[0] = 1.0f;
        return;
    }

    tensor_to_device(loss, DEVICE_CPU);
    loss->cpu_grad[0] = 1.0f;
    tensor_to_device(loss, DEVICE_GPU);
}

static void print_layer_params(const char* label, LinearLayer* layer) {
    if (layer->weight->device == DEVICE_GPU) {
        tensor_to_device(layer->weight, DEVICE_CPU);
    }
    if (layer->bias->device == DEVICE_GPU) {
        tensor_to_device(layer->bias, DEVICE_CPU);
    }

    printf("%s weight=%.6f bias=%.6f\n",
           label,
           layer->weight->cpu_data[0],
           layer->bias->cpu_data[0]);
}

static void run_training(DeviceType device, const char* label) {
    int input_shape[2] = {4, 1};
    int target_shape[2] = {4, 1};
    float input_values[4] = {0.0f, 1.0f, 2.0f, 3.0f};
    float target_values[4] = {1.0f, 3.0f, 5.0f, 7.0f};

    srand(42);

    Tensor* inputs = make_tensor(input_shape, 2, input_values, false, device);
    Tensor* targets = make_tensor(target_shape, 2, target_values, false, device);
    LinearLayer* layer = create_linear_layer(1, 1, device);

    if (inputs == NULL || targets == NULL || layer == NULL) {
        fprintf(stderr, "Failed to set up %s training run.\n", label);
        exit(1);
    }

    Tensor* parameters[] = {layer->weight, layer->bias};
    SGD* optimizer = sgd_create(parameters, 2, 0.05f);
    if (optimizer == NULL) {
        fprintf(stderr, "Failed to create optimizer for %s training run.\n", label);
        exit(1);
    }

    printf("[%s] training start\n", label);

    for (int epoch = 0; epoch < 200; epoch++) {
        Tensor* prediction = linear_forward(layer, inputs);
        Tensor* loss = tensor_mse(prediction, targets);

        seed_loss_grad(loss);
        backward(loss);
        sgd_step(optimizer);
        sgd_zero_grad(optimizer);

        if (epoch % 50 == 0 || epoch == 199) {
            float loss_value = tensor_scalar_value(loss);
            printf("[%s] epoch %d loss=%.6f\n", label, epoch, loss_value);
        }

        free_graph(loss);
    }

    print_layer_params(label, layer);

    sgd_free(optimizer);
    free_linear_layer(layer);
    free_tensor(inputs);
    free_tensor(targets);
}

int main(void) {
    printf("--- SmallGrad Training Demo ---\n\n");

    run_training(DEVICE_CPU, "CPU");
    printf("\n");
    run_training(DEVICE_GPU, "GPU");

    printf("\nTraining demo completed successfully!\n");
    return 0;
}