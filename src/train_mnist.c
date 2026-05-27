#include <stdio.h>
#include <stdlib.h>
// Notice: Pure C! No <cuda_runtime.h>, no timespec structs, no complicated math.
#include "../include/tensor.h"
#include "../include/ops.h"
#include "../include/autograd.h"
#include "../include/nn.h"
#include "../include/optim.h"
#include "../include/dataset.h"


void run_simple_training(DeviceType device, const char* label, Tensor* images, Tensor* labels) {
    int sample_count = 60000; 
    int epochs = 50;
    int batch_size = 128;
    int input_features = images->shape[1];
    int num_classes = labels->shape[1];

    // Initialize MLP and Optimizer
    int architecture[] = {input_features, 128, 64, num_classes};
    MLP* model = create_mlp(architecture, 4, device);
    if (!model) return;

    int num_params;
    Tensor** params = mlp_get_parameters(model, &num_params);
    SGD* optimizer = sgd_create(params, num_params, 0.05f);

    printf("\n[%s] Starting training on %d samples...\n", label, sample_count);

    // Core Training Loop
    for (int epoch = 0; epoch < epochs; epoch++) {
        float total_loss = 0.0f;
        int num_batches = sample_count / batch_size;

        for (int b = 0; b < num_batches; b++) {
            int start_row = b * batch_size;

            // Create zero-copy views for the current batch
            Tensor* batch_inputs = tensor_slice_view(images, start_row, batch_size);
            Tensor* batch_labels = tensor_slice_view(labels, start_row, batch_size);

            // Forward Pass
            Tensor* predictions = mlp_forward(model, batch_inputs);
            Tensor* loss = tensor_cross_entropy(predictions, batch_labels);

            // Add to total loss for logging
            total_loss += tensor_scalar_value(loss);

            // Backward Pass & Step
            seed_loss_grad(loss);
            backward(loss);
            sgd_step(optimizer);
            sgd_zero_grad(optimizer);

            // Garbage Collection
            free_graph(loss);
            free_tensor(batch_inputs);
            free_tensor(batch_labels);
        }

        printf("[%s] Epoch %d/%d | Average Loss: %.4f\n", label, epoch + 1, epochs, total_loss / num_batches);
    }

    // Clean up layer parameters
    sgd_free(optimizer);
    free(params);
    free_mlp(model);
}

int main(void) {
    printf("--- Simple SmallGrad MNIST Demo ---\n\n");

    printf("Loading MNIST data...\n");
    Tensor* all_images = load_mnist_images("data/train-images-idx3-ubyte");
    Tensor* all_labels = load_mnist_labels("data/train-labels-idx1-ubyte");

    if (all_images == NULL || all_labels == NULL) {
        fprintf(stderr, "Error: failed to load dataset.\n");
        return 1;
    }

    // 1. Run the loop on the CPU
    //run_simple_training(DEVICE_CPU, "CPU", all_images, all_labels);

    // 2. Transfer the master dataset to the GPU
    printf("\nTransferring dataset to GPU VRAM...\n");
    tensor_to_device(all_images, DEVICE_GPU);
    tensor_to_device(all_labels, DEVICE_GPU);

    // 3. Run the exact same loop on the GPU
    run_simple_training(DEVICE_GPU, "GPU", all_images, all_labels);

    // Clean up global data
    free_tensor(all_images);
    free_tensor(all_labels);

    printf("\nDemo completed successfully!\n");
    return 0;
}