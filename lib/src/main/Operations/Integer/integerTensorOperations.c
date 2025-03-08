#include <stdlib.h>

#include "Error/exceptions.h"
#include "Tensor/tensor.h"
#include "Operations/integerTensorOperations.h"


void IntegerTensor_multiply(const IntegerTensor* a, const IntegerTensor* b, const IntegerTensor* destination) {
    if (destination == NULL) {
        (void)throwIllegalArgumentException("Destination must not be NULL!");
    }

    if (a->base->dimensions != b->base->dimensions) {
        (void)throwIllegalArgumentException("Can't operate on different shaped tensors!");
    } else if (a->base->dimensions != destination->base->dimensions) {
        (void)throwIllegalArgumentException("Result tensor must have the same shape as input tensors.");
    }

    for (int i = 0; i < a->base->dimensions; i++) {
        if (a->base->shape[i] != b->base->shape[i]) {
            (void)throwIllegalArgumentException("To perform multiplication the tensor shapes must be equal.");
        } else if (a->base->shape[i] != destination->base->shape[i]) {
            (void)throwIllegalArgumentException("Result tensor must have the same shape as input tensors.");
        }
    }

    const int* data_a = a->tensor;
    const int* data_b = b->tensor;
    int* dest = destination->tensor;

    for (size_t dataIndex = 0; dataIndex < a->base->dataPoints; dataIndex++) {
        dest[dataIndex] = data_a[dataIndex] * data_b[dataIndex];
    }
}