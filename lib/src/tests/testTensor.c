#include <stdlib.h>
#include <stdio.h>

#include "testSuite.h"
#include "globals.h"
#include "Tensor/tensor.h"
#include "Operations/integerTensorOperations.h"

void testIntegerTensor() {
    int shape[4] = {16, 32, 16, 64};
    int dimensions = 4;

    IntegerTensor* tensor = createIntegerTensor(dimensions, shape);

    testSuite_assertEquals(tensor->base->dimensions, dimensions);
    freeIntegerTensor(tensor);
}

void testTensorMultiply_001() {
    const int N = 16;
    int shape[1] = {N};
    int dimensions = 1;

    IntegerTensor* tensor_a = createIntegerTensor(dimensions, shape);
    IntegerTensor* tensor_b = createIntegerTensor(dimensions, shape);
    IntegerTensor* tensor_c = createIntegerTensor(dimensions, shape);

    for (int i = 0; i < N; i++) {
        tensor_a->tensor[i] = i;
        tensor_b->tensor[i] = N - i;
    }

    IntegerTensor_multiply(tensor_a, tensor_b, tensor_c);

    for (int i = 0; i < N; i++) {
        testSuite_assertEquals(tensor_c->tensor[i], (i * (N - i)));
    }

    freeIntegerTensor(tensor_a);
    freeIntegerTensor(tensor_b);
    freeIntegerTensor(tensor_c);
}

int main() {
    ENV_UNIT_TESTING = 1;
    testIntegerTensor();
    testTensorMultiply_001();
}