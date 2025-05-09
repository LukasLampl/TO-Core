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
#include <time.h>

#include "Tests/testTensorOperations.h"
#include "Tensor/tensor.h"
#include "Operations/baseOperations.h"

#include "testSuite.h"

void profileTensorMultiply_001() {
    int N = 1920 * 1080 * 3 * 24;
    int shape[4] = {24, 1920, 1080, 3};
    int dimensions = 4;

    IntegerTensor* tensor_a = IntegerTensor_zeros(dimensions, shape);
    IntegerTensor* tensor_b = IntegerTensor_zeros(dimensions, shape);
    IntegerTensor* tensor_c = IntegerTensor_zeros(dimensions, shape);

    for (int i = 0; i < N; i++) {
        tensor_a->data[i] = i;
        tensor_b->data[i] = N - i;
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
        testSuite_assertEquals(tensor_c->data[i], (i * (N - i)));
        result += tensor_c->data[i];
    }

    printf("> Sum: %ld\n", result);

    freeIntegerTensor(tensor_a);
    freeIntegerTensor(tensor_b);
    freeIntegerTensor(tensor_c);
}

void profileTensorAdd_001() {
    int N = 1920 * 1080 * 3 * 24;
    int shape[4] = {24, 1920, 1080, 3};
    int dimensions = 4;

    IntegerTensor* tensor_a = IntegerTensor_zeros(dimensions, shape);
    IntegerTensor* tensor_b = IntegerTensor_zeros(dimensions, shape);
    IntegerTensor* tensor_c = IntegerTensor_zeros(dimensions, shape);

    for (int i = 0; i < N; i++) {
        tensor_a->data[i] = i;
        tensor_b->data[i] = N - i;
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
        testSuite_assertEquals(tensor_c->data[i], (i + (N - i)));
        result += tensor_c->data[i];
    }

    printf("> Sum: %ld\n", result);

    freeIntegerTensor(tensor_a);
    freeIntegerTensor(tensor_b);
    freeIntegerTensor(tensor_c);
}

void profileTensorDivide_001() {
    int N = 1920 * 1080 * 3 * 24;
    int shape[4] = {24, 1920, 1080, 3};
    int dimensions = 4;

    IntegerTensor* tensor_a = IntegerTensor_zeros(dimensions, shape);
    IntegerTensor* tensor_b = IntegerTensor_zeros(dimensions, shape);
    IntegerTensor* tensor_c = IntegerTensor_zeros(dimensions, shape);

    for (int i = 0; i < N; i++) {
        tensor_a->data[i] = i;
        tensor_b->data[i] = N - i;
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
        testSuite_assertEquals(tensor_c->data[i], (i / (N - i)));
        result += tensor_c->data[i];
    }

    printf("> Sum: %ld\n", result);

    freeIntegerTensor(tensor_a);
    freeIntegerTensor(tensor_b);
    freeIntegerTensor(tensor_c);
}

void profileTensorSubtract_001() {
    int N = 1920 * 1080 * 3 * 24;
    int shape[4] = {24, 1920, 1080, 3};
    int dimensions = 4;

    IntegerTensor* tensor_a = IntegerTensor_zeros(dimensions, shape);
    IntegerTensor* tensor_b = IntegerTensor_zeros(dimensions, shape);
    IntegerTensor* tensor_c = IntegerTensor_zeros(dimensions, shape);

    for (int i = 0; i < N; i++) {
        tensor_a->data[i] = i;
        tensor_b->data[i] = N - i;
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
        testSuite_assertEquals(tensor_c->data[i], (i - (N - i)));
        result += tensor_c->data[i];
    }

    printf("> Sum: %ld\n", result);

    freeIntegerTensor(tensor_a);
    freeIntegerTensor(tensor_b);
    freeIntegerTensor(tensor_c);
}