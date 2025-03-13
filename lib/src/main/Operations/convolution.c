#include "Error/exceptions.h"
#include "Tensor/tensor.h"

void IntegerTensor_convolve1D(const IntegerTensor* tensor,
    const IntegerTensor* kernel, const IntegerTensor* destination,
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

void IntegerTensor_convolve2D(const IntegerTensor* tensor,
    const IntegerTensor* kernel, const IntegerTensor* destination,
    const int stride) {
    if (stride <= 0) {
        (void)throwIllegalArgumentException("Stride must be a positive integer.");
        return;
    } else if (tensor == NULL || kernel == NULL || destination == NULL) {
        (void)throwNullPointerException("For convolution 2D the tensor, kernel and destination should not be NULL!");
        return;
    } else if (tensor->base->dimensions < 2 || kernel->base->dimensions < 2) {
        (void)throwIllegalArgumentException("For convolution 2D a 2D tensor is needed.");
        return;
    }
    
    const int kernelWidth = kernel->base->shape[kernel->base->dimensions - 2];
    const int kernelHeight = kernel->base->shape[kernel->base->dimensions - 1];
    const int tensorWidth = tensor->base->shape[tensor->base->dimensions - 2];
    const int tensorHeight = tensor->base->shape[tensor->base->dimensions - 1];
    const int destWidth = tensor->base->shape[tensor->base->dimensions - 2];
    const int outputWidth = (tensorWidth - kernelWidth) / stride + 1;
    const int outputHeight = (tensorHeight - kernelHeight) / stride + 1;

    if (destination->base->shape[destination->base->dimensions - 2] < outputHeight
        || destination->base->shape[destination->base->dimensions - 1] < outputWidth) {
        (void)throwIllegalArgumentException("Destination tensor is too small for the convolution result.");
        return;
    }

    for (int y = 0, destPtr = 0; (y + kernelHeight) <= tensorHeight; y += stride) {
        const int newDestPtr = (y + 1) * destWidth;
        
        for (int x = 0; (x + kernelWidth) <= tensorWidth; x += stride) {
            int dotProduct = 0;
    
            for (int ky = 0; ky < kernelHeight; ky++) {
                int t_yOff = (y + ky) * tensorWidth;
                int k_yOff = ky * kernelWidth;
    
                for (int kx = 0; kx < kernelWidth; kx++) {
                    dotProduct += tensor->tensor[x + kx + t_yOff] * kernel->tensor[kx + k_yOff];
                }
            }
    
            destination->tensor[destPtr++] = dotProduct;
        }

        destPtr = newDestPtr;
    }
}