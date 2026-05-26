#include "../include/nn.h"

static float random_float(float limit) {
    float unit = (float)rand() / (float)RAND_MAX;
    return (unit * 2.0f - 1.0f) * limit;
}


LinearLayer* create_linear_layer(int in_features, int out_features, DeviceType device) {
    // create layer with xavier uniform initialization for weights and bias
    LinearLayer* layer = (LinearLayer*)malloc(sizeof(LinearLayer));
    if (layer == NULL) {
        fprintf(stderr, "Error: failed to allocate memory for linear layer.\n");
        return NULL;
    }

    int weight_shape[] = {in_features, out_features};
    int bias_shape[] = {1, out_features};

    layer->weight = create_tensor(weight_shape, 2, DEVICE_CPU, true);
    if (layer->weight == NULL) {
        fprintf(stderr, "Error: problem creating weight tensor for linear layer.\n");
        free(layer);
        return NULL;
    }

    layer->bias = create_tensor(bias_shape, 2, DEVICE_CPU, true);
    if (layer->bias == NULL) {
        fprintf(stderr, "Error: problem creating bias tensor for linear layer.\n");
        free(layer);
        return NULL;
    }

    // Xavier uniform initialization for weights and biases
    
    float limit = sqrtf(6.0f / (float)(in_features + out_features));
    for (int i = 0; i < layer->weight->size; i++) {
        layer->weight->cpu_data[i] = random_float(limit);
    }
    for (int i = 0; i < layer->bias->size; i++) {
        layer->bias->cpu_data[i] = 0.01f;
    }
    if (device == DEVICE_GPU) {
        tensor_to_device(layer->weight, DEVICE_GPU);
        tensor_to_device(layer->bias, DEVICE_GPU);
    }

    return layer;

}

Tensor* linear_forward(LinearLayer* layer, Tensor* input) {
    // perform the forward pass through the linear layer and return the output tensor
    Tensor* output = tensor_matmul(input, layer->weight); // output = input @ weight

    Tensor* result = tensor_add_bias(output, layer->bias); // result = output + bias
    
    // keep the intermediate node alive for use in backward
    return result;
}

void free_linear_layer(LinearLayer* layer) {
    if (layer != NULL) {
        free_tensor(layer->weight);
        free_tensor(layer->bias);
        free(layer);
    }
}