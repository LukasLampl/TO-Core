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
#include <stdlib.h>
#include <string.h>

#include "Tensor/tensor.h"
#include "Error/exceptions.h"

/**
 * Checks whether the given dimension is a positive integer
 * and whether the given shape contains positive integers only as well.
 * 
 * @param dimension     The dimension to check.
 * @param *shape        The shape to check.
 */
void checkDimensionAndShape(const int dimensions, const int *shape) {
    if (dimensions <= 0) {
        (void)throwIllegalArgumentException("Dimensions must be a positive integer!");
    } else if (shape == NULL) {
        (void)throwNullPointerException("Shape must not be NULL!");
    }

    for (int i = 0; i < dimensions; i++) {
        if (shape[i] <= 0) {
            (void)throwIllegalArgumentException("Shape values must be positive integers!");
        }
    }
}

/**
 * Calculates the number of datapoints to expect in a tensor for the
 * given dimensions and shape.
 * 
 * @param dimensions    Number of dimensions.
 * @param *shape        Shape of the expected tensor.
 * 
 * @return The number of datapoints / data elements.
 */
int countNumberOfDataIndexes(const int dimensions, const int *shape) {
    int numberOfDataIndexes = 1;

    for (int i = 0; i < dimensions; i++) {
        numberOfDataIndexes *= shape[i];
    }

    return numberOfDataIndexes;
}

/**
 * Creates a base tensor with the given parameters.
 * 
 * @param dimensions    The number of dimensions of the tensor.
 * @param *shape        Shape of the tensor, with the size of each dimension.
 *
 * @return A base tensor pointer that can be used as metadata for actual tensors.
 */
Tensor* createTensor(const int dimensions, const int *shape) {
    (void)checkDimensionAndShape(dimensions, shape);
    const int numberOfDataIndexes = (int)countNumberOfDataIndexes(dimensions, shape);

    Tensor* tensor = (Tensor*)calloc(1, sizeof(Tensor));
    int* shape_copy = (int*)malloc(dimensions * sizeof(int));

    if (tensor == NULL || shape_copy == NULL) {
        if (tensor) (void)free(tensor);
        if (shape_copy) (void)free(shape_copy);
        (void)throwMemoryAllocationException("An error occured while trying to allocate memory for a tensor.");
    }

    (void)memcpy(shape_copy, shape, dimensions * sizeof(int));
    tensor->shape = shape_copy;
    tensor->dimensions = dimensions;
    tensor->dataPoints = numberOfDataIndexes;
    return tensor;
}

/**
 * Creates an integer based tensor with the given parameters.
 * All elements will be `0`.
 * 
 * @param dimensions    The number of dimensions of the tensor.
 * @param *shape        Shape of the tensor, with the size of each dimension.
 * 
 * @return An IntegerTensor pointer with the metadata in `tensor->base` and
 * data in `tensor->tensor`.
 */
IntegerTensor* createIntegerTensor(const int dimensions, const int *shape) {
    Tensor* base = (Tensor*)createTensor(dimensions, shape);
    IntegerTensor* tensor = (IntegerTensor*)calloc(1, sizeof(IntegerTensor));
    int* data = (int*)calloc(base->dataPoints, sizeof(int));

    if (tensor == NULL || data == NULL) {
        if (tensor) (void)free(tensor);
        if (data) (void)free(data);
        (void)throwMemoryAllocationException("An error occured while trying to allocate memory for a tensor.");
    }

    tensor->tensor = data;
    tensor->base = base;
    return tensor;
}

/**
 * Creates an float based tensor with the given parameters.
 * All elements will be `0`.
 * 
 * @param dimensions    The number of dimensions of the tensor.
 * @param *shape        Shape of the tensor, with the size of each dimension.
 * 
 * @return An FloatTensor pointer with the metadata in `tensor->base` and
 * data in `tensor->tensor`.
 */
FloatTensor* createFloatTensor(const int dimensions, const int *shape) {
    Tensor* base = (Tensor*)createTensor(dimensions, shape);
    FloatTensor* tensor = (FloatTensor*)calloc(1, sizeof(FloatTensor));
    float* data = (float*)calloc(base->dataPoints, sizeof(float));

    if (tensor == NULL || data == NULL) {
        if (tensor) (void)free(tensor);
        if (data) (void)free(data);
        (void)throwMemoryAllocationException("An error occured while trying to allocate memory for a tensor.");
    }

    tensor->tensor = data;
    tensor->base = base;
    return tensor;
}

/**
 * Creates an double based tensor with the given parameters.
 * All elements will be `0`.
 * 
 * @param dimensions    The number of dimensions of the tensor.
 * @param *shape        Shape of the tensor, with the size of each dimension.
 * 
 * @return An DoubleTensor pointer with the metadata in `tensor->base` and
 * data in `tensor->tensor`.
 */
DoubleTensor* createDoubleTensor(const int dimensions, const int *shape) {
    Tensor* base = (Tensor*)createTensor(dimensions, shape);
    DoubleTensor* tensor = (DoubleTensor*)calloc(1, sizeof(DoubleTensor));
    double* data = (double*)calloc(base->dataPoints, sizeof(double));

    if (tensor == NULL || data == NULL) {
        if (tensor) (void)free(tensor);
        if (data) (void)free(data);
        (void)throwMemoryAllocationException("An error occured while trying to allocate memory for a tensor.");
    }

    tensor->tensor = data;
    tensor->base = base;
    return tensor;
}

/**
 * Prints the data of a given Tensor.
 */
void Tensor_print(const Tensor* tensor) {
    (void)printf("Tensor: %p\n", (void*)tensor);
    (void)printf(" > Dimensions: %d\n", tensor->dimensions);
    (void)printf(" > Elements: %ld\n", tensor->dataPoints);
    (void)printf(" > Shape: [");

    for (int i = 0; i < tensor->dimensions; i++) {
        (void)printf("%d", tensor->shape[i]);

        if (i + 1 < tensor->dimensions) {
            (void)printf(", ");
        } else {
            (void)printf("]\n");
        }
    }
}

/**
 * Prints the data of an IntegerTensor recursively.
 */
void IntegerTensor_printTensor(const IntegerTensor* tensor, const int dim,
    const int ptr, const int* jumpTable) {
    if (dim >= tensor->base->dimensions - 1) {
        const int width = tensor->base->shape[tensor->base->dimensions - 1];
        (void)printf("[");

        for (int i = 0; i < width; i++) {
            (void)printf("%d", tensor->tensor[ptr + i]);
            
            if (i + 1 < width) {
                (void)printf(", ");
            }
        }

        (void)printf("]");
    } else {
        (void)printf("[");
        const int dimSize = tensor->base->shape[dim];
        const int offsetTillNextDim = jumpTable[dim];

        for (int i = 0; i < dimSize; i++) {
            int newPtr = ptr + (i * offsetTillNextDim);
            (void)IntegerTensor_printTensor(tensor, dim + 1, newPtr, jumpTable);

            if (i + 1 < dimSize) {
                (void)printf(", ");
            }
        }

        (void)printf("]");
    }

    if (dim == 0) {
        (void)printf("\n");
    }
}

/**
 * Generates a jump table for each dimension.
 * 
 * <p><b>The result:</b><br>
 * The resulting output will be an integer array that can be access at
 * any index `i` and provides the number of data to skip, until the next
 * dimension would start.
 * </p>
 * 
 * @param *tensor   The tensor from which to get the jump table from.
 * 
 * @return A pointer to an integer array with the sizes of each dimension.
 */
int *generateDimensionBasedCummulativeJumpTable(const Tensor* tensor) {
    if (tensor->dimensions <= 0) {
        (void)throwIllegalArgumentException("Tensor must have a dimension of a positive integer.");
        return NULL;
    }

    int* jumpTable = (int*)calloc(tensor->dimensions, sizeof(int));

    if (jumpTable == NULL) {
        (void)throwMemoryAllocationException("Error on allocating memory for jump table (convolution).");
        return NULL;
    }

    for (int i = tensor->dimensions - 1; i >= 0; i--) {
        jumpTable[i] = i == (tensor->dimensions - 1) ?
                            1 : jumpTable[i + 1] * tensor->shape[i + 1];
    }

    return jumpTable;
}

/**
 * Prints the given IntegerTensor.
 * 
 * @param *tensor   Tensor to print.
 */
void IntegerTensor_print(const IntegerTensor* tensor) {
    (void)Tensor_print(tensor->base);
    int* jumpTable = generateDimensionBasedCummulativeJumpTable(tensor->base);

    if (jumpTable == NULL) {
        return;
    }

    (void)IntegerTensor_printTensor(tensor, 0, 0, jumpTable);
}

/**
 * Frees a given Tensor base.
 * 
 * @param *tensor   Tensor to free.
 */
void freeTensor(Tensor* tensor) {
    if (tensor->shape != NULL) {
        (void)free(tensor->shape);
        tensor->shape = NULL;
    }

    (void)free(tensor);
}

/**
 * Frees a given IntegerTensor.
 * 
 * @param tensor    The tensor to free.
 */
void freeIntegerTensor(IntegerTensor* tensor) {
    if (tensor->base != NULL) {
        (void)freeTensor(tensor->base);
        tensor->base = NULL;
    }

    if (tensor->tensor != NULL) {
        (void)free(tensor->tensor);
        tensor->tensor = NULL;
    }

    (void)free(tensor);
}

/**
 * Frees a given FloatTensor.
 * 
 * @param tensor    The tensor to free.
 */
void freeFloatTensor(FloatTensor* tensor) {
    if (tensor->base != NULL) {
        (void)freeTensor(tensor->base);
        tensor->base = NULL;
    }

    if (tensor->tensor != NULL) {
        (void)free(tensor->tensor);
        tensor->tensor = NULL;
    }

    (void)free(tensor);
}

/**
 * Frees a given DoubleTensor.
 * 
 * @param tensor    The tensor to free.
 */
void freeDoubleTensor(DoubleTensor* tensor) {
    if (tensor->base != NULL) {
        (void)freeTensor(tensor->base);
        tensor->base = NULL;
    }

    if (tensor->tensor != NULL) {
        (void)free(tensor->tensor);
        tensor->tensor = NULL;
    }

    (void)free(tensor);
}