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

#include "Error/exceptions.h"
#include "Tensor/tensor.h"
#include "Operations/convolution.h"
#include "Network/layer.h"

#define true 1
#define false 0

/**
 * Calculates the dot product of a 1D stripe in a tensor with the given
 * kernel. This function will always use the last dimension as the measurement
 * of the "width" of the tensor and kernel.
 * 
 * @param *tensor       The tensor from which to get the dot product.
 * @param *kernel       The kernel that should be used as the multiplicant.
 * @param tensorOffset  Offset of the tensor at which to start getting the dot product.
 * @param kernelOffset  Offset of the kernels values that function as the multiplicants.
 * 
 * @return The dot product of the tensor and the kernel with the set offset.
 */
int IntegerTensor_dotProduct1D(const IntegerTensor* tensor,
    const IntegerTensor* kernel, const int tensorOffset,
    const int kernelOffset) {
    const int kernelWidth = kernel->base->shape[kernel->base->dimensions - 1];
    int dotProduct = 0;

    for (int kx = 0; kx < kernelWidth; kx++) {
        const int t_val = tensorOffset + kx >= tensor->base->dataPoints ?
                            0 : tensor->data[tensorOffset + kx];
        const int k_val = kernel->data[kernelOffset + kx];
        dotProduct += t_val * k_val;
    }

    return dotProduct;
}

/**
 * Calculates the dot product of a 1D stripe in a tensor with the given
 * kernel. This function will always use the last dimension as the measurement
 * of the "width" of the tensor and kernel.
 * 
 * @param *tensor       The tensor from which to get the dot product.
 * @param *kernel       The kernel that should be used as the multiplicant.
 * @param tensorOffset  Offset of the tensor at which to start getting the dot product.
 * @param kernelOffset  Offset of the kernels values that function as the multiplicants.
 * 
 * @return The dot product of the tensor and the kernel with the set offset.
 */
float FloatTensor_dotProduct1D(const FloatTensor* tensor,
    const FloatTensor* kernel, const int tensorOffset,
    const int kernelOffset) {
    const int kernelWidth = kernel->base->shape[kernel->base->dimensions - 1];
    float dotProduct = 0;

    for (int kx = 0; kx < kernelWidth; kx++) {
        const float t_val = tensorOffset + kx >= tensor->base->dataPoints ?
                            0 : tensor->data[tensorOffset + kx];
        const float k_val = kernel->data[kernelOffset + kx];
        dotProduct += t_val * k_val;
    }

    return dotProduct;
}

/**
 * Calculates the dot product of a 1D stripe in a tensor with the given
 * kernel. This function will always use the last dimension as the measurement
 * of the "width" of the tensor and kernel.
 * 
 * @param *tensor       The tensor from which to get the dot product.
 * @param *kernel       The kernel that should be used as the multiplicant.
 * @param tensorOffset  Offset of the tensor at which to start getting the dot product.
 * @param kernelOffset  Offset of the kernels values that function as the multiplicants.
 * 
 * @return The dot product of the tensor and the kernel with the set offset.
 */
double DoubleTensor_dotProduct1D(const DoubleTensor* tensor,
    const DoubleTensor* kernel, const int tensorOffset,
    const int kernelOffset) {
    const int kernelWidth = kernel->base->shape[kernel->base->dimensions - 1];
    double dotProduct = 0;

    for (int kx = 0; kx < kernelWidth; kx++) {
        const double t_val = tensorOffset + kx >= tensor->base->dataPoints ?
                            0 : tensor->data[tensorOffset + kx];
        const double k_val = kernel->data[kernelOffset + kx];
        dotProduct += t_val * k_val;
    }

    return dotProduct;
}

/**
 * Recursively gets the dot product of a N-Dimensional kernel and writes the
 * result to the given destination tensor.
 * 
 * <p><b>Functionality:</b><br>
 * Goes down from the highest dimension to the lowest and checks if the current
 * dimension is the first, if true, the dot product from the 1D stripe is calculated
 * and returned. If not a new recursion call happens and the process repeats until
 * the function reaches the first dimension.
 * 
 * All dot products are returned to the highest dimension, where it'll be written
 * out to the destination tensor.
 * </p>
 * 
 * @param *tensor                   Tensor that should be convolved.
 * @param *kernel                   The kernel that should be used for the convolution.
 * @param *dest                     Destination tensor in which to write the dot products.
 * @param dim                       The current dimension.
 * @param tensorPtr                 Index to the data start in the tensor.
 * @param kernelPtr                 Index to the kernel start.
 * @param *destPtr                  Pointer to the destination index.
 * @param *tensorDimJumpTable       Cummulative jump table with the entries of the offsets of each tensor dimension.
 * @param *kernelJumpTable          Cummulative jump table with the entries of the offsets of each kernel dimension.
 * @param highesternelDimension     Whether the call is from the highest dimension or not.
 * 
 * @return The dot product of the current dimension.
 */
int IntegerTensor_convolve_kernelDotProduct(const IntegerTensor* tensor,
    const IntegerTensor* kernel, const IntegerTensor* dest, const int dim,
    const int tensorPtr, const int kernelPtr, int* destPtr,
    const int* tensorDimJumpTable, const int* kernelJumpTable, const int highestKernelDimension) {
    int dotProduct = 0;

    if (dim + 1 >= kernel->base->dimensions) {
        dotProduct = (int)IntegerTensor_dotProduct1D(tensor, kernel, tensorPtr, kernelPtr);

        if (highestKernelDimension == true) {
            dest->data[(*destPtr)++] = dotProduct;
        }

        return dotProduct;
    }

    const int k_size = kernel->base->shape[dim];
    const int k_off = kernelJumpTable[dim];
    const int t_off = tensorDimJumpTable[dim];
    
    for (int i = 0; i < k_size; i++) {
        int newKernelPtr = i * k_off + kernelPtr;
        int newTensorPtr = i * t_off + tensorPtr;

        dotProduct += (int)IntegerTensor_convolve_kernelDotProduct(tensor,
            kernel, dest, dim + 1, newTensorPtr, newKernelPtr, destPtr,
            tensorDimJumpTable, kernelJumpTable, false);
    }

    if (highestKernelDimension == true) {
        dest->data[(*destPtr)++] = dotProduct;
    }

    return dotProduct;
}

/**
 * Recursively gets the dot product of a N-Dimensional kernel and writes the
 * result to the given destination tensor.
 * 
 * @param *tensor                   Tensor that should be convolved.
 * @param *kernel                   The kernel that should be used for the convolution.
 * @param *dest                     Destination tensor in which to write the dot products.
 * @param dim                       The current dimension.
 * @param tensorPtr                 Index to the data start in the tensor.
 * @param kernelPtr                 Index to the kernel start.
 * @param *destPtr                  Pointer to the destination index.
 * @param *tensorDimJumpTable       Cummulative jump table with the entries of the offsets of each tensor dimension.
 * @param *kernelJumpTable          Cummulative jump table with the entries of the offsets of each kernel dimension.
 * @param highesternelDimension     Whether the call is from the highest dimension or not.
 * 
 * @return The dot product of the current dimension.
 * 
 * @see #IntegerTensor_convolve_kernelDotProduct(const IntegerTensor* tensor,
    const IntegerTensor* kernel, const IntegerTensor* dest, const int dim,
    const int tensorPtr, const int kernelPtr, int* destPtr,
    const int* tensorDimJumpTable, const int* kernelJumpTable, const int highestKernelDimension)
 */
float FloatTensor_convolve_kernelDotProduct(const FloatTensor* tensor,
    const FloatTensor* kernel, const FloatTensor* dest, const int dim,
    const int tensorPtr, const int kernelPtr, int* destPtr,
    const int* tensorDimJumpTable, const int* kernelJumpTable, const int highestKernelDimension) {
    float dotProduct = 0;

    if (dim + 1 >= kernel->base->dimensions) {
        dotProduct = (float)FloatTensor_dotProduct1D(tensor, kernel, tensorPtr, kernelPtr);

        if (highestKernelDimension == true) {
            dest->data[(*destPtr)++] = dotProduct;
        }

        return dotProduct;
    }

    const int k_size = kernel->base->shape[dim];
    const int k_off = kernelJumpTable[dim];
    const int t_off = tensorDimJumpTable[dim];
    
    for (int i = 0; i < k_size; i++) {
        int newKernelPtr = i * k_off + kernelPtr;
        int newTensorPtr = i * t_off + tensorPtr;

        dotProduct += (float)FloatTensor_convolve_kernelDotProduct(tensor,
            kernel, dest, dim + 1, newTensorPtr, newKernelPtr, destPtr,
            tensorDimJumpTable, kernelJumpTable, false);
    }

    if (highestKernelDimension == true) {
        dest->data[(*destPtr)++] = dotProduct;
    }

    return dotProduct;
}

/**
 * Recursively gets the dot product of a N-Dimensional kernel and writes the
 * result to the given destination tensor.
 * 
 * @param *tensor                   Tensor that should be convolved.
 * @param *kernel                   The kernel that should be used for the convolution.
 * @param *dest                     Destination tensor in which to write the dot products.
 * @param dim                       The current dimension.
 * @param tensorPtr                 Index to the data start in the tensor.
 * @param kernelPtr                 Index to the kernel start.
 * @param *destPtr                  Pointer to the destination index.
 * @param *tensorDimJumpTable       Cummulative jump table with the entries of the offsets of each tensor dimension.
 * @param *kernelJumpTable          Cummulative jump table with the entries of the offsets of each kernel dimension.
 * @param highesternelDimension     Whether the call is from the highest dimension or not.
 * 
 * @return The dot product of the current dimension.
 * 
 * @see #IntegerTensor_convolve_kernelDotProduct(const IntegerTensor* tensor,
    const IntegerTensor* kernel, const IntegerTensor* dest, const int dim,
    const int tensorPtr, const int kernelPtr, int* destPtr,
    const int* tensorDimJumpTable, const int* kernelJumpTable, const int highestKernelDimension)
 */
double DoubleTensor_convolve_kernelDotProduct(const DoubleTensor* tensor,
    const DoubleTensor* kernel, const DoubleTensor* dest, const int dim,
    const int tensorPtr, const int kernelPtr, int* destPtr,
    const int* tensorDimJumpTable, const int* kernelJumpTable, const int highestKernelDimension) {
    double dotProduct = 0;

    if (dim + 1 >= kernel->base->dimensions) {
        dotProduct = (double)DoubleTensor_dotProduct1D(tensor, kernel, tensorPtr, kernelPtr);

        if (highestKernelDimension == true) {
            dest->data[(*destPtr)++] = dotProduct;
        }

        return dotProduct;
    }

    const int k_size = kernel->base->shape[dim];
    const int k_off = kernelJumpTable[dim];
    const int t_off = tensorDimJumpTable[dim];
    
    for (int i = 0; i < k_size; i++) {
        int newKernelPtr = i * k_off + kernelPtr;
        int newTensorPtr = i * t_off + tensorPtr;

        dotProduct += (double)DoubleTensor_convolve_kernelDotProduct(tensor,
            kernel, dest, dim + 1, newTensorPtr, newKernelPtr, destPtr,
            tensorDimJumpTable, kernelJumpTable, false);
    }

    if (highestKernelDimension == true) {
        dest->data[(*destPtr)++] = dotProduct;
    }

    return dotProduct;
}

/**
 * Recursively moves the given kernel around the given tensor and gets the dot product of
 * the current position.
 * 
 * <p><b>Functionality:</b><br>
 * Goes down from the highest dimension to the lowest and checks if the current
 * dimension is the first, if true, the dot product from the current location is calculated.
 * If not a new recursion call happens and the process repeats until
 * the function reaches the first dimension.
 * </p>
 * 
 * @param *tensorData           The actual tensor. (IntegerTensor, FloatTensor, DoubleTensor)
 * @param *kernelData           The actual kernel.
 * @param *destData             The actual destination tensor.
 * @param *tensorBase           The metadata of the tensor.
 * @param *kernelBase           The metadata of the kernel.
 * @param *destBase             The metadata of the destination tensor.
 * @param tensorType            Determines the types of the tensors used for convolution. (INTEGER, FLOAT, DOUBLE)
 * @param dim                   The current dimension of the recursion. (Depth of the recursion)
 * @param stride                Stride of the kernel.
 * @param tensorPtr             Index of the current tensor index at the current dimension and position of the kernel.
 * @param *destPtr              Pointer to the destination index.
 * @param *tensorDimJumpTable   Jump table where at a given index `i` the number of elements to skip, till the next dim is present (for tensor).
 * @param *kernelJumpTable      Jump table where at a given index `i` the number of elements to skip, till the next dim is present (for kernel).
 * @param highestDim            Whether the recursion is in the highest dimension or not.
 * 
 * @throw IllegalArgumentException - When the destination size at the dimension is to small.
 */
void convolve_moveKernel(const void* tensorData, const void* kernelData, const void* destData,
    const Tensor* tensorBase, const Tensor* kernelBase, const Tensor* destBase,
    const TensorType tensorType, const int dim,
    const int stride, const int tensorPtr,
    int* destPtr, const int* tensorDimJumpTable, const int* kernelJumpTable,
    const int highestDim) {
    if (dim >= tensorBase->dimensions) {
        switch (tensorType) {
        case _TENSOR_TYPE_INTEGER_:
            (void)IntegerTensor_convolve_kernelDotProduct((const IntegerTensor*)tensorData,
                (const IntegerTensor*)kernelData, (const IntegerTensor*)destData,
                0, tensorPtr, 0, destPtr, tensorDimJumpTable, kernelJumpTable, true);
            break;
        case _TENSOR_TYPE_FLOAT_:
            (void)FloatTensor_convolve_kernelDotProduct((const FloatTensor*)tensorData,
                (const FloatTensor*)kernelData, (const FloatTensor*)destData,
                0, tensorPtr, 0, destPtr, tensorDimJumpTable, kernelJumpTable, true);
            break;
        case _TENSOR_TYPE_DOUBLE_:
            (void)DoubleTensor_convolve_kernelDotProduct((const DoubleTensor*)tensorData,
                (const DoubleTensor*)kernelData, (const DoubleTensor*)destData,
                0, tensorPtr, 0, destPtr, tensorDimJumpTable, kernelJumpTable, true);
            break;
        }
        
        return;
    }

    const int t_size = tensorBase->shape[dim];
    const int k_size = kernelBase->shape[dim];
    const int d_size = destBase->shape[dim];
    const int tensorDimOff = tensorDimJumpTable[dim];
    const int min_dest_size = (t_size - k_size) / stride + 1;
    const int nextDim = dim + 1;

    if (d_size < min_dest_size) {
        (void)throwIllegalArgumentException("The destination tensor is smaller than allowed!");
        return;
    }

    for (int i = 0; (i + k_size) <= t_size; i += stride) {
        int innerTensorPtr = nextDim >= tensorBase->dimensions ?
                            tensorPtr + i
                            : i * tensorDimOff + tensorPtr;
        (void)convolve_moveKernel(tensorData, kernelData, destData,
            tensorBase, kernelBase, destBase, tensorType,
            nextDim, stride, innerTensorPtr,
            destPtr, tensorDimJumpTable, kernelJumpTable, false);
    }
}

/**
 * Executes a N-Dimensional convolution on a given tensor and kernel.
 * 
 * @param *tensor       Tensor to convolve.
 * @param *kernel       Kernel to use.
 * @param *dest         Destination tensor in which to write the results.
 * @param stride        Stride of the kernel.
 * @param tensorType    Datatype type of the tensor data (INTEGER, FLOAT, DOUBLE)
 * 
 * @throw IllegalArgumentException - When the dimensions of the tensor and kernel mismatch.
 * @throw IllegalArgumentException - When the destination size at the dimension is to small.
 * @throw NullPointerException - When either the tensor, kernel or the destination is `NULL`.
 */
void convolve(const void* tensor, const void* kernel, const void* dest,
    const int stride, const TensorType tensorType) {
    if (tensor == NULL || kernel == NULL || dest == NULL) {
        (void)throwNullPointerException("No tensor is allowed to be NULL at a convolution.");
        return;
    }

    const Tensor* tensorBase = (Tensor*)getTensorBaseByType(tensor, tensorType);
    const Tensor* kernelBase = (Tensor*)getTensorBaseByType(kernel, tensorType);
    const Tensor* destBase = (Tensor*)getTensorBaseByType(dest, tensorType);

    if (tensorBase->dimensions != kernelBase->dimensions) {
        (void)throwIllegalArgumentException("Convolution is only allowed for equal dimensional tensors.");
        return;
    }

    int destPtr = 0;
    int* tensorJumpTable = (int*)generateDimensionBasedCummulativeJumpTable(tensorBase);
    int* kernelJumpTable = (int*)generateDimensionBasedCummulativeJumpTable(kernelBase);

    if (tensorJumpTable != NULL && kernelJumpTable != NULL) {
        (void)convolve_moveKernel(tensor, kernel, dest, tensorBase, kernelBase,
            destBase, tensorType, 0, stride, 0, &destPtr,
            tensorJumpTable, kernelJumpTable, true);
    }
    
    if (tensorJumpTable != NULL) (void)free(tensorJumpTable);
    if (kernelJumpTable != NULL) (void)free(kernelJumpTable);
}

/**
 * Executes a N-Dimensional convolution on a given tensor and kernel.
 * 
 * @param *tensor   Tensor to convolve.
 * @param *kernel   Kernel to use.
 * @param *dest     Destination tensor in which to write the results.
 * @param stride    Stride of the kernel.
 * 
 * @throw IllegalArgumentException - When the dimensions of the tensor and kernel mismatch.
 * @throw IllegalArgumentException - When the destination size at the dimension is to small.
 * @throw NullPointerException - When either the tensor, kernel or the destination is `NULL`.
 */
void IntegerTensor_convolve(const IntegerTensor* tensor,
    const IntegerTensor* kernel, const IntegerTensor* dest, const int stride) {
    (void)convolve(tensor, kernel, dest, stride, _TENSOR_TYPE_INTEGER_);
}

/**
 * Executes a N-Dimensional convolution on a given tensor and kernel.
 * 
 * @param *tensor   Tensor to convolve.
 * @param *kernel   Kernel to use.
 * @param *dest     Destination tensor in which to write the results.
 * @param stride    Stride of the kernel.
 * 
 * @throw IllegalArgumentException - When the dimensions of the tensor and kernel mismatch.
 * @throw IllegalArgumentException - When the destination size at the dimension is to small.
 * @throw NullPointerException - When either the tensor, kernel or the destination is `NULL`.
 */
void FloatTensor_convolve(const FloatTensor* tensor,
    const FloatTensor* kernel, const FloatTensor* dest, const int stride) {
    (void)convolve(tensor, kernel, dest, stride, _TENSOR_TYPE_FLOAT_);
}

/**
 * Executes a N-Dimensional convolution on a given tensor and kernel.
 * 
 * @param *tensor   Tensor to convolve.
 * @param *kernel   Kernel to use.
 * @param *dest     Destination tensor in which to write the results.
 * @param stride    Stride of the kernel.
 * 
 * @throw IllegalArgumentException - When the dimensions of the tensor and kernel mismatch.
 * @throw IllegalArgumentException - When the destination size at the dimension is to small.
 * @throw NullPointerException - When either the tensor, kernel or the destination is `NULL`.
 */
void DoubleTensor_convolve(const DoubleTensor* tensor,
    const DoubleTensor* kernel, const DoubleTensor* dest, const int stride) {
    (void)convolve(tensor, kernel, dest, stride, _TENSOR_TYPE_DOUBLE_);
}

/**
 * Creates a ConvolutionLayer based on the given parameters.
 * 
 * @param *kernel       Kernel to use for the convolution.
 * @param *destination  Optional destination of the convolution values.
 * @param stride        Stride of the convolution.
 * @param tensorType    Type of the tensors involved (all must be equal).
 * 
 * @throws NullPointerException - When the given kernel is `NULL`.
 * @throws IllegalArgumentException - When the stride is not a positive integer.
 */
ConvolutionLayer* createConvolutionLayer(const void* kernel, const void* destination,
    const int stride, const TensorType tensorType) {
    if (kernel == NULL) {
        (void)throwNullPointerException("Kernel of convolution must not be NULL!");
        return NULL;
    } else if (stride <= 0) {
        (void)throwIllegalArgumentException("Stride must be a positive integer.");
        return NULL;
    }

    Layer* base = (Layer*)createLayer(tensorType);
    ConvolutionLayer* layer = (ConvolutionLayer*)calloc(1, sizeof(ConvolutionLayer));

    if (base == NULL || layer == NULL) {
        if (base != NULL) (void)free(base);
        if (layer != NULL) (void)free(layer);
        (void)throwMemoryAllocationException("While trying to generate ConvolutionLayer.");
        return NULL;
    }

    layer->base = base;
    layer->kernel = kernel;
    layer->stride = stride;
    layer->destination = destination;
    layer->isDestinationSet = destination == NULL ? false : true;
    return layer;
}

/**
 * Creates a ConvolutionLayer with the given kernel, destination and stride.
 * 
 * <p><b>Note:</b><br>
 * The destination must not be initialized and can be set to `NULL`. When set
 * to `NULL` the used Network will generate the destination automatically.
 * </p>
 * 
 * @param *kernel       The kernel to use for the convolution.
 * @param *destination  Optional destination to which to write the results.
 * @param stride        Stride of the convolution.
 */
ConvolutionLayer* Integer_createConvolutionLayer(const IntegerTensor* kernel,
    const IntegerTensor* destination, const int stride) {
    return (ConvolutionLayer*)createConvolutionLayer(kernel,
        destination, stride, _TENSOR_TYPE_INTEGER_);
}

/**
 * Creates a ConvolutionLayer with the given kernel, destination and stride.
 * 
 * <p><b>Note:</b><br>
 * The destination must not be initialized and can be set to `NULL`. When set
 * to `NULL` the used Network will generate the destination automatically.
 * </p>
 * 
 * @param *kernel       The kernel to use for the convolution.
 * @param *destination  Optional destination to which to write the results.
 * @param stride        Stride of the convolution.
 */
ConvolutionLayer* Float_createConvolutionLayer(const FloatTensor* kernel,
    const FloatTensor* destination, const int stride) {
    return (ConvolutionLayer*)createConvolutionLayer(kernel,
        destination, stride, _TENSOR_TYPE_FLOAT_);
}

/**
 * Creates a ConvolutionLayer with the given kernel, destination and stride.
 * 
 * <p><b>Note:</b><br>
 * The destination must not be initialized and can be set to `NULL`. When set
 * to `NULL` the used Network will generate the destination automatically.
 * </p>
 * 
 * @param *kernel       The kernel to use for the convolution.
 * @param *destination  Optional destination to which to write the results.
 * @param stride        Stride of the convolution.
 */
ConvolutionLayer* Double_createConvolutionLayer(const DoubleTensor* kernel,
    const DoubleTensor* destination, const int stride) {
    return (ConvolutionLayer*)createConvolutionLayer(kernel,
        destination, stride, _TENSOR_TYPE_DOUBLE_);
}

/**
 * Executes the convolution with the given parameters of the ConvolutionLayer
 * on the given input. The result is written into the destination tensor of
 * the given ConvolutionLayer.
 * 
 * <p><b>Warning:</b><br>
 * The type of the input is determined by the layer type. This mean if the layer type
 * if `IntegerTensor`, the input should also be an `IntegerTensor` or else undefined
 * behaviour will occur.
 * </p>
 * 
 * @param *layer    The ConvolutionLayer with all parameters for the convolution.
 * @param *input    Pointer to the input that should be convolved.
 * 
 * @throws IllegalArgumentException - When the ConvolutionLayer has no fixed destination,
 * but the destination pointer is `NULL`.
 */
void ConvolutionLayer_forward(const ConvolutionLayer* layer, void* input) {
    if (layer->isDestinationSet == false && layer->destination == NULL) {
        (void)throwIllegalArgumentException("No destination is prohibited.");
        return;
    }

    switch (layer->base->inputType) {
    case _TENSOR_TYPE_INTEGER_:
        (void)IntegerTensor_convolve((IntegerTensor*)input, (IntegerTensor*)layer->kernel,
            (IntegerTensor*)layer->destination, layer->stride);
        break;
    case _TENSOR_TYPE_FLOAT_:
        (void)FloatTensor_convolve((FloatTensor*)input, (FloatTensor*)layer->kernel,
                (FloatTensor*)layer->destination, layer->stride);
        break;
    case _TENSOR_TYPE_DOUBLE_:
        (void)DoubleTensor_convolve((DoubleTensor*)input, (DoubleTensor*)layer->kernel,
                (DoubleTensor*)layer->destination, layer->stride);
        break;
    }
}

/**
 * Frees a given ConvolutionLayer.
 * 
 * @param *layer    ConvolutionLayer to free.
 */
void ConvolutionLayer_free(ConvolutionLayer* layer) {
    if (layer == NULL) {
        return;
    }

    (void)freeLayer(layer->base);
    (void)free(layer);
}