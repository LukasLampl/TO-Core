#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "Tensor/tensor.h"
#include "Error/exceptions.h"

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

int countNumberOfDataIndexes(const int dimensions, const int *shape) {
    int numberOfDataIndexes = 1;

    for (int i = 0; i < dimensions; i++) {
        numberOfDataIndexes *= shape[i];
    }

    return numberOfDataIndexes;
}

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

void IntegerTensor_printTensor(const IntegerTensor* tensor, const int dim,
    const int ptr) {
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

        for (int i = 0; i < dimSize; i++) {
            int newPtr = ptr + (i * dimSize);
            (void)IntegerTensor_printTensor(tensor, dim + 1, newPtr);

            if (i + 1 < dimSize) {
                (void)printf(", ");
            }
        }

        (void)printf("]");
    }
}

void IntegerTensor_print(const IntegerTensor* tensor) {
    (void)Tensor_print(tensor->base);
    (void)IntegerTensor_printTensor(tensor, 0, 0);
}

void freeTensor(Tensor* tensor) {
    if (tensor->shape != NULL) {
        (void)free(tensor->shape);
        tensor->shape = NULL;
    }

    (void)free(tensor);
}

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