#include "../include/nn.h"


// helper for xavier uniform initialization
static float random_float(float limit) {
    float unit = (float)rand() / (float)RAND_MAX;
    return (unit * 2.0f - 1.0f) * limit;
}


// ------------ Linear Layer -----------------

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


// ----------- Convolutional layer ---------------

Conv2dLayer* create_conv2d_layer(int in_channels, int out_channels, int filter_size, int stride, int padding, DeviceType device) {
    // create convolutional layer given input parameters

    Conv2dLayer* layer = (Conv2dLayer*)malloc(sizeof(Conv2dLayer));
    if (layer == NULL) {
        fprintf(stderr, "Error: problem with allocating memory for convolutional layer initialization.\n");
        return NULL;
    }
    
    // initialize parameters
    layer->in_channels = in_channels;
    layer->out_channels = out_channels;
    layer->filter_size = filter_size;
    layer->stride = stride;
    layer->padding = padding;

    // [number of filters, channels per filter, filter_height, filter_width]
    int weight_shape[] = {out_channels, in_channels, filter_size, filter_size};

    int bias_shape[] = {1, out_channels, 1, 1};

    layer->weight = create_tensor(weight_shape, 4, DEVICE_CPU, true);
    if (layer->weight == NULL) {
        fprintf(stderr, "Error: problem creating weight tensor for convolutional layer weights.\n");
        free(layer);
        return NULL;
    }

    layer->bias = create_tensor(bias_shape, 4, DEVICE_CPU, true);
    if (layer->bias == NULL) {
        fprintf(stderr, "Error: problem creating weight tensor for convolutional layer bias.\n");
        free(layer);
        return NULL;
    }

    // xavier uniform initialization

    int filter_area = filter_size * filter_size;
    float filter_in = filter_area * in_channels; // in_features
    float filter_out = filter_area * out_channels; // out_features
    float limit = sqrtf(6.0f / (float) (filter_in + filter_out));

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


void free_conv2d_layer(Conv2dLayer* layer) {
    // free all memory allocated for the convolutional layer
    if (layer != NULL) {
        free_tensor(layer->weight);
        free_tensor(layer->bias);
        free(layer);
    }
}

Tensor* conv_forward(Conv2dLayer* layer, Tensor* input) {

    
}

// --------- MLP ----------

MLP* create_mlp(int* architecture, int num_layers, DeviceType device) {
    MLP* model = (MLP*)malloc(sizeof(MLP));
    if (model == NULL) {
        fprintf(stderr, "Error: failed to allocate memory for the MLP struct.\n");
        return NULL;
    }
    model->dev = device;
    model->num_layers = num_layers - 1;
    model->layers = (LinearLayer**)malloc(model->num_layers * sizeof(LinearLayer*));
    if (model->layers == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for MLP layers array.\n");
        free(model);
        return NULL;
    }

    for (int i = 0; i < model->num_layers; i++) {
        model->layers[i] = create_linear_layer(architecture[i], architecture[i + 1], device);
    }
    return model;
}

Tensor* mlp_forward(MLP* model, Tensor* input) {
    Tensor* current = input;
    for (int i = 0; i < model->num_layers; i++) {
        current = linear_forward(model->layers[i], current);
        // apply relu to hidden layers
        if (i < model->num_layers - 1) {
            current = tensor_relu(current);
        }
    }

    return current;
}   

Tensor** mlp_get_parameters(MLP* model, int* out_num_parameters) {
    *out_num_parameters = model->num_layers * 2; // each layer has weight and bias
    Tensor** params = (Tensor**)malloc(*out_num_parameters * sizeof(Tensor*));
    if (params == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for MLP parameters array.\n");
        return NULL;
    }

    for (int i = 0; i < model->num_layers; i++) {
        params[i * 2] = model->layers[i]->weight;
        params[i * 2 + 1] = model->layers[i]->bias;
    }
    return params;
}

void free_mlp(MLP* model) {
    if (model != NULL) {
        for (int i = 0; i < model->num_layers; i++) {
            free_linear_layer(model->layers[i]);
        }
        free(model->layers);
        free(model);
    }
}