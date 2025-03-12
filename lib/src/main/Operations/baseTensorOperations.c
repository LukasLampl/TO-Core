#include <stdlib.h>

#include "Tensor/tensor.h"
#include "Error/exceptions.h"

int getElementIndex(const Tensor* tensor, const int* indices) {
    int index = 0;
    int stride = 1;

    for (int i = tensor->dimensions - 1; i >= 0; i--) {
        index += indices[i] * stride;
        stride *= tensor->shape[i];
    }

    return index;
}

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

void IntegerTensor_multiply(const IntegerTensor* a, const IntegerTensor* b, const IntegerTensor* destination) {
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

void IntegerTensor_scalarMultiply(const IntegerTensor* a, const int scalar, const IntegerTensor* destination) {
    (void)checkTensorCompatability(a, destination);

    const int* data_a = a->tensor;
    int* dest = destination->tensor;
    const int* endPtr = data_a + a->base->dataPoints;

    while (data_a < endPtr) {
        *dest++ = scalar * (*data_a++);
    }
}