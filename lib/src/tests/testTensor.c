#include <stdlib.h>
#include <stdio.h>
#include <time.h>

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

void testTensorMultiply_002() {
    int N = 1920 * 1080 * 3 * 24;
    int shape[4] = {24, 1920, 1080, 3};
    int dimensions = 4;

    IntegerTensor* tensor_a = createIntegerTensor(dimensions, shape);
    IntegerTensor* tensor_b = createIntegerTensor(dimensions, shape);
    IntegerTensor* tensor_c = createIntegerTensor(dimensions, shape);

    for (int i = 0; i < N; i++) {
        tensor_a->tensor[i] = i;
        tensor_b->tensor[i] = N - i;
    }

    clock_t start = clock();
    IntegerTensor_multiply(tensor_a, tensor_b, tensor_c);
    clock_t end = clock();
    double totalTime = ((double)(end - start) / (double)CLOCKS_PER_SEC);

    printf("\nReport for %d multiplications (int).\n", N);
    printf("============================================\n");
    printf("> Time took: %f s\n", totalTime);
    printf("> Throughput: %f multiplications per second.\n", ((double)N / totalTime));
    size_t result = 0;

    for (int i = 0; i < N; i++) {
        testSuite_assertEquals(tensor_c->tensor[i], (i * (N - i)));
        result += tensor_c->tensor[i];
    }

    printf("> Sum: %ld\n", result);

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

void testTensorAdd_002() {
    int N = 1920 * 1080 * 3 * 24;
    int shape[4] = {24, 1920, 1080, 3};
    int dimensions = 4;

    IntegerTensor* tensor_a = createIntegerTensor(dimensions, shape);
    IntegerTensor* tensor_b = createIntegerTensor(dimensions, shape);
    IntegerTensor* tensor_c = createIntegerTensor(dimensions, shape);

    for (int i = 0; i < N; i++) {
        tensor_a->tensor[i] = i;
        tensor_b->tensor[i] = N - i;
    }

    clock_t start = clock();
    IntegerTensor_add(tensor_a, tensor_b, tensor_c);
    clock_t end = clock();
    double totalTime = ((double)(end - start) / (double)CLOCKS_PER_SEC);

    printf("\nReport for %d additions (int).\n", N);
    printf("============================================\n");
    printf("> Time took: %f s\n", totalTime);
    printf("> Throughput: %f additions per second.\n", ((double)N / totalTime));
    size_t result = 0;

    for (int i = 0; i < N; i++) {
        testSuite_assertEquals(tensor_c->tensor[i], (i + (N - i)));
        result += tensor_c->tensor[i];
    }

    printf("> Sum: %ld\n", result);

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

void testTensorDivide_002() {
    int N = 1920 * 1080 * 3 * 24;
    int shape[4] = {24, 1920, 1080, 3};
    int dimensions = 4;

    IntegerTensor* tensor_a = createIntegerTensor(dimensions, shape);
    IntegerTensor* tensor_b = createIntegerTensor(dimensions, shape);
    IntegerTensor* tensor_c = createIntegerTensor(dimensions, shape);

    for (int i = 0; i < N; i++) {
        tensor_a->tensor[i] = i;
        tensor_b->tensor[i] = N - i;
    }

    clock_t start = clock();
    IntegerTensor_divide(tensor_a, tensor_b, tensor_c);
    clock_t end = clock();
    double totalTime = ((double)(end - start) / (double)CLOCKS_PER_SEC);

    printf("\nReport for %d divisions (int).\n", N);
    printf("============================================\n");
    printf("> Time took: %f s\n", totalTime);
    printf("> Throughput: %f divisions per second.\n", ((double)N / totalTime));
    size_t result = 0;

    for (int i = 0; i < N; i++) {
        testSuite_assertEquals(tensor_c->tensor[i], (i / (N - i)));
        result += tensor_c->tensor[i];
    }

    printf("> Sum: %ld\n", result);

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

void testTensorSubtract_002() {
    int N = 1920 * 1080 * 3 * 24;
    int shape[4] = {24, 1920, 1080, 3};
    int dimensions = 4;

    IntegerTensor* tensor_a = createIntegerTensor(dimensions, shape);
    IntegerTensor* tensor_b = createIntegerTensor(dimensions, shape);
    IntegerTensor* tensor_c = createIntegerTensor(dimensions, shape);

    for (int i = 0; i < N; i++) {
        tensor_a->tensor[i] = i;
        tensor_b->tensor[i] = N - i;
    }

    clock_t start = clock();
    IntegerTensor_subtract(tensor_a, tensor_b, tensor_c);
    clock_t end = clock();
    double totalTime = ((double)(end - start) / (double)CLOCKS_PER_SEC);

    printf("\nReport for %d subtractions (int).\n", N);
    printf("============================================\n");
    printf("> Time took: %f s\n", totalTime);
    printf("> Throughput: %f subtractions per second.\n", ((double)N / totalTime));
    size_t result = 0;

    for (int i = 0; i < N; i++) {
        testSuite_assertEquals(tensor_c->tensor[i], (i - (N - i)));
        result += tensor_c->tensor[i];
    }

    printf("> Sum: %ld\n", result);

    freeIntegerTensor(tensor_a);
    freeIntegerTensor(tensor_b);
    freeIntegerTensor(tensor_c);
}

int main() {
    ENV_UNIT_TESTING = 1;
    testIntegerTensor();
    testTensorMultiply_001();
    testTensorMultiply_002();

    testTensorAdd_001();
    testTensorAdd_002();

    testTensorDivide_001();
    testTensorDivide_002();

    testTensorSubtract_001();
    testTensorSubtract_002();
}