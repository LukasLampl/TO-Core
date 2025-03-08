#include <stdlib.h>

#include "Tensor/tensor.h"

int getElementIndex(const Tensor* tensor, const int* indices) {
    int index = 0;
    int stride = 1;

    for (int i = tensor->dimensions - 1; i >= 0; i--) {
        index += indices[i] * stride;
        stride *= tensor->shape[i];
    }

    return index;
}
