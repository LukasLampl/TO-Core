#include <stdio.h>
#include <stdlib.h>

#include "Tests/Integer/testTensorOperations.h"
#include "Tensor/tensor.h"
#include "Operations/tensorOperations.h"

#include "testSuite.h"

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

void testTensorAdd_001() {
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

    IntegerTensor_add(tensor_a, tensor_b, tensor_c);

    for (int i = 0; i < N; i++) {
        testSuite_assertEquals(tensor_c->tensor[i], (i + (N - i)));
    }

    freeIntegerTensor(tensor_a);
    freeIntegerTensor(tensor_b);
    freeIntegerTensor(tensor_c);
}

void testTensorDivide_001() {
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

    IntegerTensor_divide(tensor_a, tensor_b, tensor_c);

    for (int i = 0; i < N; i++) {
        testSuite_assertEquals(tensor_c->tensor[i], (i / (N - i)));
    }

    freeIntegerTensor(tensor_a);
    freeIntegerTensor(tensor_b);
    freeIntegerTensor(tensor_c);
}

void testTensorSubtract_001() {
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

    IntegerTensor_subtract(tensor_a, tensor_b, tensor_c);

    for (int i = 0; i < N; i++) {
        testSuite_assertEquals(tensor_c->tensor[i], (i - (N - i)));
    }

    freeIntegerTensor(tensor_a);
    freeIntegerTensor(tensor_b);
    freeIntegerTensor(tensor_c);
}