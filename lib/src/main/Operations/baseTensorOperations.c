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

#include <stdlib.h>

#include "Tensor/tensor.h"
#include "Error/exceptions.h"

/**
 * Returns the element index in the given tensors data for
 * a given indeces array.
 * 
 * <p><b>Example:</b><br>
 * If you want to find the index of [3, 4, 5], you'll need to
 * put in the according tensor and an integer array with the
 * indeces that you're searching for.
 * </p>
 * 
 * @param tensor    The tensor from which to get the index from.
 * @param *indices  Desired indices in the tensor.
 * 
 * @return The element index in the tensors data array.
 */
int getElementIndex(const Tensor* tensor, const int* indices) {
    int index = 0;
    int stride = 1;

    for (int i = tensor->dimensions - 1; i >= 0; i--) {
        index += indices[i] * stride;
        stride *= tensor->shape[i];
    }

    return index;
}

/**
 * Checks whether the dimensions of the given tensors match
 * and whether the shapes match as well.
 * 
 * @param *a    First tensor to check for compatability.
 * @param *b    Second tensor to check against the first.
 * 
 * @throws IllegalArgumentException - When either the dimension does not match
 * or the shapes are different.
 */
void checkTensorCompatability(const IntegerTensor* a, const IntegerTensor* b) {
    if (a->base->dimensions != b->base->dimensions) {
        (void)throwIllegalArgumentException("Can't operate on different shaped tensors!");
    }

    for (int i = 0; i < a->base->dimensions; i++) {
        if (a->base->shape[i] != b->base->shape[i]) {
            (void)throwIllegalArgumentException("To perform multiplication the tensor shapes must be equal.");
        }
    }
}

/**
 * Multiplies two given tensors and writes the results to a given
 * destination tensor.
 * 
 * <p><b>Important:</b><br>
 * This function only allows multiplication of two identically
 * shaped tensors.
 * </p>
 * 
 * @param *a            Pointer to the multiplier tensor.
 * @param *b            Pointer to the multiplicand tensor.
 * @param *destination  Pointer to the destination tensor in which to write.
 */
void IntegerTensor_multiply(const IntegerTensor* a, const IntegerTensor* b,
    const IntegerTensor* destination) {
    (void)checkTensorCompatability(a, b);
    (void)checkTensorCompatability(a, destination);

    const int* data_a = a->tensor;
    const int* data_b = b->tensor;
    int* dest = destination->tensor;
    const int* endPtr = data_a + a->base->dataPoints;

    while (data_a < endPtr) {
        *dest++ = (*data_a++) * (*data_b++);
    }
}

/**
 * Divides two given tensors and writes the results to a given
 * destination tensor.
 * 
 * <p><b>Important:</b><br>
 * This function only allows division of two identically
 * shaped tensors.
 * </p>
 * 
 * @param *a            Pointer to the dividend tensor.
 * @param *b            Pointer to the divisor tensor.
 * @param *destination  Pointer to the destination tensor in which to write.
 */
void IntegerTensor_divide(const IntegerTensor* a, const IntegerTensor* b, const IntegerTensor* destination) {
    (void)checkTensorCompatability(a, b);
    (void)checkTensorCompatability(a, destination);

    const int* data_a = a->tensor;
    const int* data_b = b->tensor;
    int* dest = destination->tensor;
    const int* endPtr = data_a + a->base->dataPoints;

    while (data_a < endPtr) {
        *dest++ = (*data_a++) / (*data_b++);
    }
}

/**
 * Adds two given tensors and writes the results to a given
 * destination tensor.
 * 
 * <p><b>Important:</b><br>
 * This function only allows addition of two identically
 * shaped tensors.
 * </p>
 * 
 * @param *a            Pointer to the first summand tensor.
 * @param *b            Pointer to the second summand tensor.
 * @param *destination  Pointer to the destination tensor in which to write.
 */
void IntegerTensor_add(const IntegerTensor* a, const IntegerTensor* b, const IntegerTensor* destination) {
    (void)checkTensorCompatability(a, b);
    (void)checkTensorCompatability(a, destination);

    const int* data_a = a->tensor;
    const int* data_b = b->tensor;
    int* dest = destination->tensor;
    const int* endPtr = data_a + a->base->dataPoints;

    while (data_a < endPtr) {
        *dest++ = (*data_a++) + (*data_b++);
    }
}

/**
 * Subtracts two given tensors and writes the results to a given
 * destination tensor.
 * 
 * <p><b>Important:</b><br>
 * This function only allows subtraction of two identically
 * shaped tensors.
 * </p>
 * 
 * @param *a            Pointer to the minuend tensor.
 * @param *b            Pointer to the subtrahend tensor.
 * @param *destination  Pointer to the destination tensor in which to write.
 */
void IntegerTensor_subtract(const IntegerTensor* a, const IntegerTensor* b, const IntegerTensor* destination) {
    (void)checkTensorCompatability(a, b);
    (void)checkTensorCompatability(a, destination);

    const int* data_a = a->tensor;
    const int* data_b = b->tensor;
    int* dest = destination->tensor;
    const int* endPtr = data_a + a->base->dataPoints;

    while (data_a < endPtr) {
        *dest++ = (*data_a++) - (*data_b++);
    }
}

/**
 * Multiplies a given tensor by a given scalar and writes
 * the results into a given destination tensor.
 * 
 * @param *a            Pointer to the tensor to multiply.
 * @param *scalar       Scalar to multiply the tensor with.
 * @param *destination  Pointer to the destination tensor in which to write.
 */
void IntegerTensor_scalarMultiply(const IntegerTensor* a, const int scalar, const IntegerTensor* destination) {
    (void)checkTensorCompatability(a, destination);

    const int* data_a = a->tensor;
    int* dest = destination->tensor;
    const int* endPtr = data_a + a->base->dataPoints;

    while (data_a < endPtr) {
        *dest++ = scalar * (*data_a++);
    }
}