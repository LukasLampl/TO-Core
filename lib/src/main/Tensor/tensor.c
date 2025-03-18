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

enum PrintFormat {
    DECIMAL,
    FLOAT,
    DOUBLE
};

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
Tensor* createTensorBase(const int dimensions, const int *shape) {
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
 * Creates an Tensor with the given dimension, shape and data type.
 * All elements will be `0`.
 * 
 * @param dimensions    Number of dimensions the tensor should have.
 * @param *shape        Shape of the Tensor, with the size of each dimension.
 * @param tensorType    Data type of the tensor.
 * 
 * @return A pointer to the created Tensor with the metadata in `tensor->base` and
 * data in `tensor->tensor`.
 */
void* createTensor(const int dimensions, const int *shape, const TensorType tensorType) {
    size_t sizeOfTensor = 0;
    size_t sizeOfElement = 0;

    switch (tensorType) {
    case _TENSOR_TYPE_INTEGER_:
        sizeOfTensor = sizeof(IntegerTensor);
        sizeOfElement = sizeof(int);
        break;
    case _TENSOR_TYPE_FLOAT_:
        sizeOfTensor = sizeof(FloatTensor);
        sizeOfElement = sizeof(float);
        break;
    case _TENSOR_TYPE_DOUBLE_:
        sizeOfTensor = sizeof(DoubleTensor);
        sizeOfElement = sizeof(double);
        break;
    default:
        (void)throwIllegalArgumentException("The tensor type does not exist!");
        return NULL;
    }

    Tensor* base = (Tensor*)createTensorBase(dimensions, shape);
    
    // Error handling in tensor base creation.
    if (base == NULL) {
        return NULL;
    }

    void* tensor = (void*)calloc(1, sizeOfTensor);
    void* data = (void*)calloc(base->dataPoints, sizeOfElement);

    if (tensor == NULL || data == NULL) {
        if (tensor != NULL) (void)free(tensor);
        if (data != NULL) (void)free(data);
        if (base != NULL) (void)freeTensor(base);
        (void)throwMemoryAllocationException("An error occured while trying to allocate memory for a tensor.");
        return NULL;
    }

    switch (tensorType) {
    case _TENSOR_TYPE_INTEGER_:
        ((IntegerTensor*)tensor)->base = base;
        ((IntegerTensor*)tensor)->tensor = (int*)data;
        break;
    case _TENSOR_TYPE_FLOAT_:
        ((FloatTensor*)tensor)->base = base;
        ((FloatTensor*)tensor)->tensor = (float*)data;
        break;
    case _TENSOR_TYPE_DOUBLE_:
        ((DoubleTensor*)tensor)->base = base;
        ((DoubleTensor*)tensor)->tensor = (double*)data;
        break;
    default:
        (void)throwIllegalArgumentException("The tensor type does not exist!");
        return NULL;
    }

    return tensor;
}

/**
 * Sets the values of the given tensor to the given value.
 * 
 * @param *tensor       The Tensor to set.
 * @param tensorType    Type of the given Tensor.
 * @param value         The value to set to each element.
 */
void initTensorByValue(const void* tensor, const TensorType tensorType, const double value) {
    switch (tensorType) {
    case _TENSOR_TYPE_INTEGER_: {
        int* data = (int*)((IntegerTensor*)tensor)->tensor;
        const int* end = data + ((IntegerTensor*)tensor)->base->dataPoints;

        while (data < end) {
            *data++ = (int)value;
        }

        break;
    }
    case _TENSOR_TYPE_FLOAT_: {
        float* data = (float*)((FloatTensor*)tensor)->tensor;
        const float* end = data + ((FloatTensor*)tensor)->base->dataPoints;

        while (data < end) {
            *data++ = (float)value;
        }

        break;
    }
    case _TENSOR_TYPE_DOUBLE_: {
        double* data = (double*)((DoubleTensor*)tensor)->tensor;
        const double* end = data + ((DoubleTensor*)tensor)->base->dataPoints;

        while (data < end) {
            *data++ = (double)value;
        }

        break;
    }
    }
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
IntegerTensor* IntegerTensor_zeros(const int dimensions, const int *shape) {
    return (IntegerTensor*)createTensor(dimensions, shape, _TENSOR_TYPE_INTEGER_);
}

/**
 * Creates an integer based tensor with the given parameters.
 * All elements will be `1`.
 * 
 * @param dimensions    The number of dimensions of the tensor.
 * @param *shape        Shape of the tensor, with the size of each dimension.
 * 
 * @return An IntegerTensor pointer with the metadata in `tensor->base` and
 * data in `tensor->tensor`.
 */
IntegerTensor* IntegerTensor_ones(const int dimensions, const int *shape) {
    IntegerTensor* tensor = (IntegerTensor*)IntegerTensor_zeros(dimensions, shape);
    (void)initTensorByValue(tensor, _TENSOR_TYPE_INTEGER_, 1);
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
FloatTensor* FloatTensor_zeros(const int dimensions, const int *shape) {
    return (FloatTensor*)createTensor(dimensions, shape, _TENSOR_TYPE_FLOAT_);
}

/**
 * Creates a float based tensor with the given parameters.
 * All elements will be `1`.
 * 
 * @param dimensions    The number of dimensions of the tensor.
 * @param *shape        Shape of the tensor, with the size of each dimension.
 * 
 * @return An FloatTensor pointer with the metadata in `tensor->base` and
 * data in `tensor->tensor`.
 */
FloatTensor* FloatTensor_ones(const int dimensions, const int *shape) {
    FloatTensor* tensor = (FloatTensor*)FloatTensor_zeros(dimensions, shape);
    (void)initTensorByValue(tensor, _TENSOR_TYPE_FLOAT_, 1);
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
DoubleTensor* DoubleTensor_zeros(const int dimensions, const int *shape) {
    return (DoubleTensor*)createTensor(dimensions, shape, _TENSOR_TYPE_DOUBLE_);
}

/**
 * Creates a double based tensor with the given parameters.
 * All elements will be `1`.
 * 
 * @param dimensions    The number of dimensions of the tensor.
 * @param *shape        Shape of the tensor, with the size of each dimension.
 * 
 * @return An DoubleTensor pointer with the metadata in `tensor->base` and
 * data in `tensor->tensor`.
 */
DoubleTensor* DoubleTensor_ones(const int dimensions, const int *shape) {
    DoubleTensor* tensor = (DoubleTensor*)DoubleTensor_zeros(dimensions, shape);
    (void)initTensorByValue(tensor, _TENSOR_TYPE_DOUBLE_, 1);
    return tensor;
}

/**
 * Prints the data of a given Tensor.
 * 
 * @param *tensor   Tensor to print.
 */
void Tensor_printMeta(const Tensor* tensor) {
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
 * 
 * @param *tensor       Tensor to print.
 * @param *base         Base of the tensor.
 * @param dim           The current dimension beeing printed.
 * @param ptr           Index offset to the current dimension.
 * @param *jumpTable    Table containing the offsets the the next dimension.
 * @param PrintFormat   What type of data to print.
 */
void printTensor(const void* tensor, const Tensor* base, const int dim,
    const int ptr, const int* jumpTable, const enum PrintFormat format) {
    if (dim >= base->dimensions - 1) {
        const int width = base->shape[base->dimensions - 1];
        (void)printf("[");

        for (int i = 0; i < width; i++) {
            switch (format) {
            case DECIMAL:
                (void)printf("%d", ((int*)tensor)[ptr + i]);
                break;
            case FLOAT:
                (void)printf("%f", ((float*)tensor)[ptr + i]);
                break;
            case DOUBLE:
                (void)printf("%f", ((double*)tensor)[ptr + i]);
                break;
            }
            
            if (i + 1 < width) {
                (void)printf(", ");
            }
        }

        (void)printf("]");
    } else {
        (void)printf("[");
        const int dimSize = base->shape[dim];
        const int offsetTillNextDim = jumpTable[dim];

        for (int i = 0; i < dimSize; i++) {
            int newPtr = ptr + (i * offsetTillNextDim);
            (void)printTensor(tensor, base, dim + 1, newPtr, jumpTable, format);

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
    (void)Tensor_printMeta(tensor->base);
    int* jumpTable = generateDimensionBasedCummulativeJumpTable(tensor->base);

    if (jumpTable == NULL) {
        return;
    }

    (void)printTensor(tensor->tensor, tensor->base, 0, 0, jumpTable, DECIMAL);
    (void)free(jumpTable);
}

/**
 * Prints the given FloatTensor.
 * 
 * @param *tensor   Tensor to print.
 */
void FloatTensor_print(const FloatTensor* tensor) {
    (void)Tensor_printMeta(tensor->base);
    int* jumpTable = generateDimensionBasedCummulativeJumpTable(tensor->base);

    if (jumpTable == NULL) {
        return;
    }

    (void)printTensor(tensor->tensor, tensor->base, 0, 0, jumpTable, FLOAT);
    (void)free(jumpTable);
}

/**
 * Prints the given DoubleTensor.
 * 
 * @param *tensor   Tensor to print.
 */
void DoubleTensor_print(const DoubleTensor* tensor) {
    (void)Tensor_printMeta(tensor->base);
    int* jumpTable = generateDimensionBasedCummulativeJumpTable(tensor->base);

    if (jumpTable == NULL) {
        return;
    }

    (void)printTensor(tensor->tensor, tensor->base, 0, 0, jumpTable, DOUBLE);
    (void)free(jumpTable);
}