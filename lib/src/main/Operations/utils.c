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

#include <string.h>

#include "Tensor/tensor.h"
#include "Operations/utils.h"
#include "Error/exceptions.h"

/**
 * Flattens a given tensor, by resolving the shape and
 * setting the dimension to `1`. This can be used, since all
 * the data is already stored in a 1D-Array.
 * 
 * @param *tensor   Base of a tensor to flatten.
 */
void flatten(Tensor* tensor) {
    int* shape = (int*)realloc(tensor->shape, sizeof(int));

    if (shape == NULL) {
        (void)throwMemoryAllocationException("While trying to allocate memory for tensor flattening.");
        return;
    }

    shape[0] = tensor->dataPoints;
    tensor->shape = shape;
    tensor->dimensions = 1;
}

/**
 * Flattens a given tensor from N-Dimensions to 1-Dimension.
 * 
 * @param *tensor   Tensor to flatten.
 */
void IntegerTensor_flatten(const IntegerTensor* tensor) {
    (void)flatten(tensor->base);
}

/**
 * Flattens a given tensor from N-Dimensions to 1-Dimension.
 * 
 * @param *tensor   Tensor to flatten.
 */
void FloatTensor_flatten(const FloatTensor* tensor) {
    (void)flatten(tensor->base);
}

/**
 * Flattens a given tensor from N-Dimensions to 1-Dimension.
 * 
 * @param *tensor   Tensor to flatten.
 */
void DoubleTensor_flatten(const DoubleTensor* tensor) {
    (void)flatten(tensor->base);
}

/**
 * Reshapes a given tensor to the new shape and dimensions.
 * 
 * The new shape must match with the old number of datapoints, or else
 * an exception will be thrown.
 * 
 * @param *tensor       Tensor base to reshape.
 * @param *newShape     New shape of the tensor.
 * @param dimensions    The number of dimensions of the new shape.
 * 
 * @throws IllegalArgumentException - When the new number of datapoints does not match the old one.
 */
void reshape(Tensor* tensor, const int* newShape, const int dimensions) {
        size_t newDataPoints = (size_t)countNumberOfDataIndexes(dimensions, newShape);

        if (newDataPoints != tensor->dataPoints) {
            (void)throwIllegalArgumentException("Reshaping must result in equal number of datapoints.");
            return;
        }

        int* shape_copy = (int*)malloc(dimensions * sizeof(int));

        if (shape_copy == NULL) {
            (void)throwMemoryAllocationException("An error occured while trying to reshape a tensor.");
        }
    
        (void)memcpy(shape_copy, newShape, dimensions * sizeof(int));
        (void)free(tensor->shape);
        tensor->shape = shape_copy;
        tensor->dimensions = dimensions;
}

/**
 * Reshapes a given tensor to the given shape.
 * 
 * @param *tensor       Tensor to reshape.
 * @param *shape        The new shape.
 * @param dimensions    Number of dimensions in the new shape.
 * 
 * @throws IllegalArgumentException - When the number of datapoints of the new shape does not the old one.
 */
void IntegerTensor_reshape(const IntegerTensor* tensor, const int* shape, const int dimensions) {
    (void)reshape(tensor->base, shape, dimensions);
}

/**
 * Reshapes a given tensor to the given shape.
 * 
 * @param *tensor       Tensor to reshape.
 * @param *shape        The new shape.
 * @param dimensions    Number of dimensions in the new shape.
 * 
 * @throws IllegalArgumentException - When the number of datapoints of the new shape does not the old one.
 */
void FloatTensor_reshape(const FloatTensor* tensor, const int* shape, const int dimensions) {
    (void)reshape(tensor->base, shape, dimensions);
}

/**
 * Reshapes a given tensor to the given shape.
 * 
 * @param *tensor       Tensor to reshape.
 * @param *shape        The new shape.
 * @param dimensions    Number of dimensions in the new shape.
 * 
 * @throws IllegalArgumentException - When the number of datapoints of the new shape does not the old one.
 */
void DoubleTensor_reshape(const DoubleTensor* tensor, const int* shape, const int dimensions) {
    (void)reshape(tensor->base, shape, dimensions);
}