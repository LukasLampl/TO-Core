#include "Error/exceptions.h"
#include "Tensor/tensor.h"

void IntegerTensor_convolve1D(const IntegerTensor* tensor, const IntegerTensor* kernel, const IntegerTensor* destination,
    const int stride) {
    if (stride <= 0) {
        (void)throwIllegalArgumentException("Stride must be a positive integer.");
        return;
    } else if (tensor == NULL || kernel == NULL || destination == NULL) {
        (void)throwNullPointerException("For convolution 1D the tensor, kernel and destination should not be NULL!");
        return;
    }
    
    const int kernelWidth = kernel->base->shape[kernel->base->dimensions - 1];
    const int tensorWidth = tensor->base->shape[tensor->base->dimensions - 1];
    const int outputWidth = (tensorWidth - kernelWidth) / stride + 1;

    if (destination->base->shape[destination->base->dimensions - 1] < outputWidth) {
        (void)throwIllegalArgumentException("Destination tensor is too small for the convolution result.");
        return;
    }

    for (int x = 0, destPtr = 0; (x + kernelWidth) <= tensorWidth; x += stride) {
        int dotProduct = 0;
        
        for (int kx = 0; kx < kernelWidth; kx++) {
            dotProduct += tensor->tensor[x + kx] * kernel->tensor[kx];
        }

        destination->tensor[destPtr++] = dotProduct;
    }
}