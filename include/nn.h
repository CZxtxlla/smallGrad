#ifndef NN_H
#define NN_H

#include "tensor.h"
#include "ops.h"
#include <math.h>


#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    Tensor* weight; // shape [in_features, out_features]
    Tensor* bias; // shape [1, out_features]
} LinearLayer;


LinearLayer* create_linear_layer(int in_features, int out_features, DeviceType device); // xavier uniform weights and bias
Tensor* linear_forward(LinearLayer* layer, Tensor* input); // pass the input tensor through linear layer
void free_linear_layer(LinearLayer* layer); // free the memory allocated for the linear layer and its tensors

#ifdef __cplusplus
}
#endif

#endif