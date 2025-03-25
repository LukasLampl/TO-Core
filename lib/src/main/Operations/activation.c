/////////////////////////////////////////////////////////////
///////////////////////    LICENSE    ///////////////////////
/////////////////////////////////////////////////////////////
/*
The TO-Core library for basic Tensor Operations.
Copyright (C) 2025  Lukas Nian En Lampl

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

#include <math.h>

#include "Tensor/tensor.h"
#include "Operations/activation.h"
#include "mathUtils.h"
#include "Error/exceptions.h"

/**
 * Calculates the ReLU of a given data tensor and sets the results back
 * into the data tensor.
 * 
 * <p><b>Definition:</b><br>
 * ReLU(x) = `0` when `x <= 0`, but `x` when `x > 0`.
 * </p>
 */
void ReLU(void* data, const Tensor* base, const TensorType tensorType) {
    switch (tensorType) {
    case _TENSOR_TYPE_INTEGER_: {
        int* start = (int*)data;
        const int* end = start + base->dataPoints;

        while (start < end) {
            *start = (int)int_max(0, *start);
            start++;
        }
        break;
    }
    case _TENSOR_TYPE_FLOAT_: {
        float* start = (float*)data;
        const float* end = start + base->dataPoints;

        while (start < end) {
            *start = (float)float_max(0, *start);
            start++;
        }
        break;
    }
    case _TENSOR_TYPE_DOUBLE_: {
        double* start = (double*)data;
        const double* end = start + base->dataPoints;

        while (start < end) {
            *start = (double)double_max(0, *start);
            start++;
        }
        break;
    }
    }
}

/**
 * Calculates the ReLU activation function values for each
 * element of the given tensor and sets the element to the
 * result.
 * 
 * @param *tensor   Tensor to which to apply the ReLU activation function.
 */
void IntegerTensor_ReLU(IntegerTensor* tensor) {
    (void)ReLU(tensor->data, tensor->base, _TENSOR_TYPE_INTEGER_);
}

/**
 * Calculates the ReLU activation function values for each
 * element of the given tensor and sets the element to the
 * result.
 * 
 * @param *tensor   Tensor to which to apply the ReLU activation function.
 */
void FloatTensor_ReLU(FloatTensor* tensor) {
    (void)ReLU(tensor->data, tensor->base, _TENSOR_TYPE_FLOAT_);
}

/**
 * Calculates the ReLU activation function values for each
 * element of the given tensor and sets the element to the
 * result.
 * 
 * @param *tensor   Tensor to which to apply the ReLU activation function.
 */
void DoubleTensor_ReLU(DoubleTensor* tensor) {
    (void)ReLU(tensor->data, tensor->base, _TENSOR_TYPE_DOUBLE_);
}

/**
 * Calculates the Leaky ReLU of a given data tensor and sets the results back
 * into the data tensor.
 * 
 * <p><b>Definition:</b><br>
 * Leaky_ReLU(x) = `alpha * x` when `x <= 0` or `x` when `x > 0`.
 * </p>
 */
void Leaky_ReLU(void* data, const double alpha, const Tensor* base,
    const TensorType tensorType) {
    switch (tensorType) {
    case _TENSOR_TYPE_INTEGER_: {
        int* start = (int*)data;
        const int* end = start + base->dataPoints;

        while (start < end) {
            const int max = (int)int_max(0, *start);
            *start = max == 0 ? alpha * *start : max;
            start++;
        }
        break;
    }
    case _TENSOR_TYPE_FLOAT_: {
        float* start = (float*)data;
        const float* end = start + base->dataPoints;

        while (start < end) {
            const float max = (float)float_max(0, *start);
            *start = max == 0 ? alpha * *start : max;
            start++;
        }
        break;
    }
    case _TENSOR_TYPE_DOUBLE_: {
        double* start = (double*)data;
        const double* end = start + base->dataPoints;

        while (start < end) {
            const double max = (double)double_max(0, *start);
            *start = max == 0 ? alpha * *start : max;
            start++;
        }
        break;
    }
    }
}

/**
 * Calculates the Leaky ReLU activation function values for each
 * element of the given tensor and sets the element to the
 * result.
 * 
 * @param *tensor   Tensor to which to apply the Leaky ReLU activation function.
 * @param alpha     The alpha multiplier used for negative values.
 */
void IntegerTensor_LeakyReLU(IntegerTensor* tensor, const int alpha) {
    (void)Leaky_ReLU(tensor->data, alpha, tensor->base, _TENSOR_TYPE_INTEGER_);
}

/**
 * Calculates the Leaky ReLU activation function values for each
 * element of the given tensor and sets the element to the
 * result.
 * 
 * @param *tensor   Tensor to which to apply the Leaky ReLU activation function.
 * @param alpha     The alpha multiplier used for negative values.
 */
void FloatTensor_LeakyReLU(FloatTensor* tensor, const float alpha) {
    (void)Leaky_ReLU(tensor->data, alpha, tensor->base, _TENSOR_TYPE_FLOAT_);
}

/**
 * Calculates the Leaky ReLU activation function values for each
 * element of the given tensor and sets the element to the
 * result.
 * 
 * @param *tensor   Tensor to which to apply the Leaky ReLU activation function.
 * @param alpha     The alpha multiplier used for negative values.
 */
void DoubleTensor_LeakyReLU(DoubleTensor* tensor, const double alpha) {
    (void)Leaky_ReLU(tensor->data, alpha, tensor->base, _TENSOR_TYPE_DOUBLE_);
}

/**
 * Calculates the Sigmoid of a given data tensor and sets the results back
 * into the data tensor.
 * 
 * <p><b>Definition:</b><br>
 * Sigmoid(x) = `1.0 / (1.0 + e^-x)`.
 * </p>
 */
void Sigmoid(void* data, const Tensor* base, const TensorType tensorType) {
    switch (tensorType) {
    case _TENSOR_TYPE_INTEGER_: {
        int* start = (int*)data;
        const int* end = start + base->dataPoints;

        while (start < end) {
            *start = (1.0) / (1.0 + (double)exp(-*start));
            start++;
        }
        break;
    }
    case _TENSOR_TYPE_FLOAT_: {
        float* start = (float*)data;
        const float* end = start + base->dataPoints;

        while (start < end) {
            *start = (1.0) / (1.0 + (double)exp(-*start));
            start++;
        }
        break;
    }
    case _TENSOR_TYPE_DOUBLE_: {
        double* start = (double*)data;
        const double* end = start + base->dataPoints;

        while (start < end) {
            *start = (1.0) / (1.0 + (double)exp(-*start));
            start++;
        }
        break;
    }
    }
}

/**
 * Calculates the Sigmoid activation function values for each
 * element of the given tensor and sets the element to the
 * result.
 * 
 * @param *tensor   Tensor to which to apply the Sigmoid activation function.
 */
void IntegerTensor_Sigmoid(IntegerTensor* tensor) {
    (void)Sigmoid(tensor->data, tensor->base, _TENSOR_TYPE_INTEGER_);
}

/**
 * Calculates the Sigmoid activation function values for each
 * element of the given tensor and sets the element to the
 * result.
 * 
 * @param *tensor   Tensor to which to apply the Sigmoid activation function.
 */
void FloatTensor_Sigmoid(FloatTensor* tensor) {
    (void)Sigmoid(tensor->data, tensor->base, _TENSOR_TYPE_FLOAT_);
}

/**
 * Calculates the Sigmoid activation function values for each
 * element of the given tensor and sets the element to the
 * result.
 * 
 * @param *tensor   Tensor to which to apply the Sigmoid activation function.
 */
void DoubleTensor_Sigmoid(DoubleTensor* tensor) {
    (void)Sigmoid(tensor->data, tensor->base, _TENSOR_TYPE_DOUBLE_);
}

/**
 * Calculates the Tanh of a given data tensor and sets the results back
 * into the data tensor.
 * 
 * <p><b>Definition:</b><br>
 * Tanh(x) = `(e^x - e^-x) / (e^x + e^-x)`.
 * 
 * Equal to
 * 
 * Tanh(x) = `1.0 - (2.0 / (e^2x + 1))`
 * </p>
 */
void Tanh(void* data, const Tensor* base, const TensorType tensorType) {
    switch (tensorType) {
    case _TENSOR_TYPE_INTEGER_: {
        int* start = (int*)data;
        const int* end = start + base->dataPoints;

        while (start < end) {
            *start = 1.0 - (2.0 / ((double)exp(2 * *start) + 1.0));
            start++;
        }
        break;
    }
    case _TENSOR_TYPE_FLOAT_: {
        float* start = (float*)data;
        const float* end = start + base->dataPoints;

        while (start < end) {
            *start = 1.0 - (2.0 / ((double)exp(2 * *start) + 1.0));
            start++;
        }
        break;
    }
    case _TENSOR_TYPE_DOUBLE_: {
        double* start = (double*)data;
        const double* end = start + base->dataPoints;

        while (start < end) {
            *start = 1.0 - (2.0 / ((double)exp(2 * *start) + 1.0));
            start++;
        }
        break;
    }
    }
}

/**
 * Calculates the Tanh activation function values for each
 * element of the given tensor and sets the element to the
 * result.
 * 
 * @param *tensor   Tensor to which to apply the Tanh activation function.
 */
void IntegerTensor_Tanh(IntegerTensor* tensor) {
    (void)Tanh(tensor->data, tensor->base, _TENSOR_TYPE_INTEGER_);
}

/**
 * Calculates the Tanh activation function values for each
 * element of the given tensor and sets the element to the
 * result.
 * 
 * @param *tensor   Tensor to which to apply the Tanh activation function.
 */
void FloatTensor_Tanh(FloatTensor* tensor) {
    (void)Tanh(tensor->data, tensor->base, _TENSOR_TYPE_FLOAT_);
}

/**
 * Calculates the Tanh activation function values for each
 * element of the given tensor and sets the element to the
 * result.
 * 
 * @param *tensor   Tensor to which to apply the Tanh activation function.
 */
void DoubleTensor_Tanh(DoubleTensor* tensor) {
    (void)Tanh(tensor->data, tensor->base, _TENSOR_TYPE_DOUBLE_);
}

/**
 * Creates an ActivationLayer that defines the type of activation function
 * to apply to a certain input at a certain stage in a network.
 * 
 * @param activationType    The activation function to apply.
 * @param alpha             Optional alpha to use (Only available for certain functions).
 * @param tensorType        Type of the input tensor.
 */
ActivationLayer* createActivationLayer(const ActivationType activationType,
    const double alpha, const TensorType tensorType) {
    Layer* base = (Layer*)createLayer(tensorType, NULL);
    ActivationLayer* layer = (ActivationLayer*)calloc(1, sizeof(ActivationLayer));

    if (base == NULL || layer == NULL) {
        if (base != NULL) (void)free(base);
        if (layer != NULL) (void)free(layer);
        (void)throwMemoryAllocationException("While trying to generate ActivationLayer.");
        return NULL;
    }

    layer->base = base;
    layer->type = activationType;
    layer->alpha = alpha;
    return layer;
}


/**
 * Forwards a given ReLU ActivationLayer with the given input.
 * 
 * @param *layer    ActivationLayer with processing information.
 * @param *input    Input to process.
 */
void forward_ReLU(const ActivationLayer* layer, const void* input) {
    switch (layer->base->inputType) {
    case _TENSOR_TYPE_INTEGER_: {
        (void)IntegerTensor_ReLU((IntegerTensor*)input);
        break;
    }
    case _TENSOR_TYPE_FLOAT_: {
        (void)FloatTensor_ReLU((FloatTensor*)input);
        break;
    }
    case _TENSOR_TYPE_DOUBLE_: {
        (void)DoubleTensor_ReLU((DoubleTensor*)input);
        break;
    }
    }
}

/**
 * Forwards a given Leaky ReLU ActivationLayer with the given input.
 * 
 * @param *layer    ActivationLayer with processing information.
 * @param *input    Input to process.
 */
void forward_LeakyReLU(const ActivationLayer* layer, const void* input) {
    switch (layer->base->inputType) {
    case _TENSOR_TYPE_INTEGER_: {
        (void)IntegerTensor_LeakyReLU((IntegerTensor*)input, layer->alpha);
        break;
    }
    case _TENSOR_TYPE_FLOAT_: {
        (void)FloatTensor_LeakyReLU((FloatTensor*)input, layer->alpha);
        break;
    }
    case _TENSOR_TYPE_DOUBLE_: {
        (void)DoubleTensor_LeakyReLU((DoubleTensor*)input, layer->alpha);
        break;
    }
    }
}

/**
 * Forwards a given Sigmoid ActivationLayer with the given input.
 * 
 * @param *layer    ActivationLayer with processing information.
 * @param *input    Input to process.
 */
void forward_Sigmoid(const ActivationLayer* layer, const void* input) {
    switch (layer->base->inputType) {
    case _TENSOR_TYPE_INTEGER_: {
        (void)IntegerTensor_Sigmoid((IntegerTensor*)input);
        break;
    }
    case _TENSOR_TYPE_FLOAT_: {
        (void)FloatTensor_Sigmoid((FloatTensor*)input);
        break;
    }
    case _TENSOR_TYPE_DOUBLE_: {
        (void)DoubleTensor_Sigmoid((DoubleTensor*)input);
        break;
    }
    }
}

/**
 * Forwards a given Tanh ActivationLayer with the given input.
 * 
 * @param *layer    ActivationLayer with processing information.
 * @param *input    Input to process.
 */
void forward_Tanh(const ActivationLayer* layer, const void* input) {
    switch (layer->base->inputType) {
    case _TENSOR_TYPE_INTEGER_: {
        (void)IntegerTensor_Tanh((IntegerTensor*)input);
        break;
    }
    case _TENSOR_TYPE_FLOAT_: {
        (void)FloatTensor_Tanh((FloatTensor*)input);
        break;
    }
    case _TENSOR_TYPE_DOUBLE_: {
        (void)DoubleTensor_Tanh((DoubleTensor*)input);
        break;
    }
    }
}

/**
 * Creates an ActivationLayer with the given activation function.
 * 
 * @param activationType    Type of activation function to apply.
 * @param alpha             Alpha to apply, when needed (only certain functions need this).
 */
ActivationLayer* Integer_createActivationLayer(const ActivationType activationType,
    const int alpha) {
    return (ActivationLayer*)createActivationLayer(activationType, alpha, _TENSOR_TYPE_INTEGER_);
}

/**
 * Creates an ActivationLayer with the given activation function.
 * 
 * @param activationType    Type of activation function to apply.
 * @param alpha             Alpha to apply, when needed (only certain functions need this).
 */
ActivationLayer* Float_createActivationLayer(const ActivationType activationType,
    const int alpha) {
    return (ActivationLayer*)createActivationLayer(activationType, alpha, _TENSOR_TYPE_FLOAT_);
}

/**
 * Creates an ActivationLayer with the given activation function.
 * 
 * @param activationType    Type of activation function to apply.
 * @param alpha             Alpha to apply, when needed (only certain functions need this).
 */
ActivationLayer* Double_createActivationLayer(const ActivationType activationType,
    const int alpha) {
    return (ActivationLayer*)createActivationLayer(activationType, alpha, _TENSOR_TYPE_DOUBLE_);
}

/**
 * Applies the activation function on the given input.
 * 
 * <p><b>Important:</b><br>
 * The input itself is modified.
 * </p>
 * 
 * @param *layer    The ActivationLayer to apply.
 * @param *input    Input on which to apply the ActivationLayer.
 */
void ActivationLayer_forward(ActivationLayer* layer, void* input) {
    switch (layer->type) {
    case RELU:
        (void)forward_ReLU(layer, input);
        break;
    case LEAKY_RELU:
        (void)forward_LeakyReLU(layer, input);
        break;
    case SIGMOID:
        (void)forward_Sigmoid(layer, input);
        break;
    case TANH:
        (void)forward_Tanh(layer, input);
        break;
    }

    layer->base->destination = input;
}

/**
 * Frees a given ActivationLayer.
 * 
 * @param *layer    The layer to free.
 */
void ActivationLayer_free(ActivationLayer* layer) {
    if (layer == NULL) {
        return;
    }

    (void)free(layer->base);
    (void)free(layer);
}