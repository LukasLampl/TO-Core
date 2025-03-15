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

#include <stdio.h>
#include "Error/exceptions.h"
#include "Tensor/tensor.h"

#define true 1
#define false 0

/**
 * Calculates the dot product of a 1D stripe in a tensor with the given
 * kernel. This function will always use the last dimension as the measurement
 * of the "width" of the tensor and kernel.
 * 
 * <p><b>Caution:</b><br>
 * This function does not check for valid parameters, if this is needed,
 * please refer to #IntegerTensor_dotProduct_1D(IntegerTensor* tensor,
 *  IntegerTensor* kernel, int tensorOffset, int kernelOffset).
 * </p>
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
                            0 : tensor->tensor[tensorOffset + kx];
        const int k_val = kernel->tensor[kernelOffset + kx];
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
int IntegerTensor_dotProduct_1D(const IntegerTensor* tensor,
    const IntegerTensor* kernel, const int tensorOffset,
    const int kernelOffset) {
    if (tensor == NULL || kernel == NULL) {
        (void)throwNullPointerException("For dot product the kernel and tensor must not be NULL!");
        return 0;
    } else if (tensorOffset < 0 || kernelOffset < 0) {
        (void)throwIllegalArgumentException("Offset must be a positive integer.");
        return 0;
    }

    return IntegerTensor_dotProduct1D(tensor, kernel, tensorOffset, kernelOffset);
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
 * @param highesternelDimension     Whether the call is from the highest dimension or not.
 * @param *destPtr                  Pointer to the destination index.
 * @param *tensorDimJumpTable       Cummulative jump table with the entries of the offsets of each tensor dimension.
 * @param *kernelJumpTable          Cummulative jump table with the entries of the offsets of each kernel dimension.
 * 
 * @return The dot product of the current dimension.
 */
int IntegerTensor_convolve_kernelDotProduct(const IntegerTensor* tensor,
    const IntegerTensor* kernel, const IntegerTensor* dest, const int dim,
    const int tensorPtr, const int kernelPtr, const int highestKernelDimension,
    int* destPtr, const int* tensorDimJumpTable, const int* kernelJumpTable) {
    int dotProduct = 0;

    if (dim + 1 >= kernel->base->dimensions) {
        dotProduct = (int)IntegerTensor_dotProduct1D(tensor, kernel, tensorPtr, kernelPtr);

        if (highestKernelDimension == true) {
            dest->tensor[(*destPtr)++] = dotProduct;
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
            kernel, dest, dim + 1, newTensorPtr, newKernelPtr, false, destPtr, tensorDimJumpTable, kernelJumpTable);
    }

    if (highestKernelDimension == true) {
        dest->tensor[(*destPtr)++] = dotProduct;
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
 * @param *tensor                   Tensor that should be convolved.
 * @param *kernel                   The kernel that should be used for the convolution.
 * @param *dest                     Destination tensor in which to write the dot products.
 * @param dim                       The current dimension.
 * @param stride                    Stride at which the kernel moves.
 * @param tensorPtr                 Index to the data start in the tensor.
 * @param highestDim                Whether the call is from the highest dimension or not.
 * @param *destPtr                  Pointer to the destination index.
 * @param *tensorDimJumpTable       Cummulative jump table with the entries of the offsets of each tensor dimension.
 * @param *kernelJumpTable          Cummulative jump table with the entries of the offsets of each kernel dimension.
 * 
 * @throw IllegalArgumentException - When the destination size at the dimension is to small.
 */
void IntegerTensor_convolve_moveKernel(const IntegerTensor* tensor,
    const IntegerTensor* kernel, const IntegerTensor* dest, const int dim,
    const int stride, const int tensorPtr, const int highestDim,
    int* destPtr, const int* tensorDimJumpTable, const int* kernelJumpTable) {
    if (dim >= tensor->base->dimensions) {
        (void)IntegerTensor_convolve_kernelDotProduct(tensor, kernel,
                dest, 0, tensorPtr, 0, true, destPtr, tensorDimJumpTable, kernelJumpTable);
        return;
    }

    const int t_size = tensor->base->shape[dim];
    const int k_size = kernel->base->shape[dim];
    const int d_size = dest->base->shape[dim];
    const int tensorDimOff = tensorDimJumpTable[dim];
    const int min_dest_size = (t_size - k_size) / stride + 1;
    const int nextDim = dim + 1;

    if (d_size < min_dest_size) {
        (void)throwIllegalArgumentException("The destination tensor is smaller than allowed!");
        return;
    }

    for (int i = 0; (i + k_size) <= t_size; i += stride) {
        int innerTensorPtr = nextDim >= tensor->base->dimensions ?
                            tensorPtr + i
                            : i * tensorDimOff + tensorPtr;
        (void)IntegerTensor_convolve_moveKernel(tensor,
            kernel, dest, nextDim, stride, innerTensorPtr,
            false, destPtr, tensorDimJumpTable, kernelJumpTable);
    }
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
    if (tensor == NULL || kernel == NULL || dest == NULL) {
        (void)throwNullPointerException("No tensor is allowed to be NULL at a convolution.");
        return;
    } else if (tensor->base->dimensions != kernel->base->dimensions) {
        (void)throwIllegalArgumentException("Convolution is only allowed for equal dimensional tensors.");
        return;
    }

    int destPtr = 0;
    int* tensorJumpTable = (int*)generateDimensionBasedCummulativeJumpTable(tensor->base);
    int* kernelJumpTable = (int*)generateDimensionBasedCummulativeJumpTable(kernel->base);

    if (tensorJumpTable == NULL || kernelJumpTable == NULL) {
        (void)free(tensorJumpTable);
        (void)free(kernelJumpTable);
        return;
    }

    (void)IntegerTensor_convolve_moveKernel(tensor, kernel, dest,
        0, stride, 0, true, &destPtr, tensorJumpTable, kernelJumpTable);
    (void)free(tensorJumpTable);
    (void)free(kernelJumpTable);
}