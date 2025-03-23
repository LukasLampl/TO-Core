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

#include "Tensor/tensor.h"
#include "Operations/baseOperations.h"

#include "Tests/testTensorOperations.h"
#include "testSuite.h"

void testTensorMultiply_001() {
    const int N = 16;
    int shape[1] = {N};
    int dimensions = 1;

    IntegerTensor* tensor_a = IntegerTensor_zeros(dimensions, shape);
    IntegerTensor* tensor_b = IntegerTensor_zeros(dimensions, shape);
    IntegerTensor* tensor_c = IntegerTensor_zeros(dimensions, shape);

    for (int i = 0; i < N; i++) {
        tensor_a->data[i] = i;
        tensor_b->data[i] = N - i;
    }

    IntegerTensor_multiply(tensor_a, tensor_b, tensor_c);

    for (int i = 0; i < N; i++) {
        testSuite_assertEquals(tensor_c->data[i], (i * (N - i)));
    }

    freeIntegerTensor(tensor_a);
    freeIntegerTensor(tensor_b);
    freeIntegerTensor(tensor_c);
}

void testTensorMultiply_002() {
    const int N = 16;
    int shape[1] = {N};
    int dimensions = 1;

    FloatTensor* tensor_a = FloatTensor_zeros(dimensions, shape);
    FloatTensor* tensor_b = FloatTensor_zeros(dimensions, shape);
    FloatTensor* tensor_c = FloatTensor_zeros(dimensions, shape);

    for (int i = 0; i < N; i++) {
        tensor_a->data[i] = i;
        tensor_b->data[i] = N - i;
    }

    FloatTensor_multiply(tensor_a, tensor_b, tensor_c);

    for (int i = 0; i < N; i++) {
        testSuite_assertEquals(tensor_c->data[i], (i * (N - i)));
    }

    freeFloatTensor(tensor_a);
    freeFloatTensor(tensor_b);
    freeFloatTensor(tensor_c);
}

void testTensorAdd_001() {
    const int N = 16;
    int shape[1] = {N};
    int dimensions = 1;

    IntegerTensor* tensor_a = IntegerTensor_zeros(dimensions, shape);
    IntegerTensor* tensor_b = IntegerTensor_zeros(dimensions, shape);
    IntegerTensor* tensor_c = IntegerTensor_zeros(dimensions, shape);

    for (int i = 0; i < N; i++) {
        tensor_a->data[i] = i;
        tensor_b->data[i] = N - i;
    }

    IntegerTensor_add(tensor_a, tensor_b, tensor_c);

    for (int i = 0; i < N; i++) {
        testSuite_assertEquals(tensor_c->data[i], (i + (N - i)));
    }

    freeIntegerTensor(tensor_a);
    freeIntegerTensor(tensor_b);
    freeIntegerTensor(tensor_c);
}

void testTensorDivide_001() {
    const int N = 16;
    int shape[1] = {N};
    int dimensions = 1;

    IntegerTensor* tensor_a = IntegerTensor_zeros(dimensions, shape);
    IntegerTensor* tensor_b = IntegerTensor_zeros(dimensions, shape);
    IntegerTensor* tensor_c = IntegerTensor_zeros(dimensions, shape);

    for (int i = 0; i < N; i++) {
        tensor_a->data[i] = i;
        tensor_b->data[i] = N - i;
    }

    IntegerTensor_divide(tensor_a, tensor_b, tensor_c);

    for (int i = 0; i < N; i++) {
        testSuite_assertEquals(tensor_c->data[i], (i / (N - i)));
    }

    freeIntegerTensor(tensor_a);
    freeIntegerTensor(tensor_b);
    freeIntegerTensor(tensor_c);
}

void testTensorSubtract_001() {
    const int N = 16;
    int shape[1] = {N};
    int dimensions = 1;

    IntegerTensor* tensor_a = IntegerTensor_zeros(dimensions, shape);
    IntegerTensor* tensor_b = IntegerTensor_zeros(dimensions, shape);
    IntegerTensor* tensor_c = IntegerTensor_zeros(dimensions, shape);

    for (int i = 0; i < N; i++) {
        tensor_a->data[i] = i;
        tensor_b->data[i] = N - i;
    }

    IntegerTensor_subtract(tensor_a, tensor_b, tensor_c);

    for (int i = 0; i < N; i++) {
        testSuite_assertEquals(tensor_c->data[i], (i - (N - i)));
    }

    freeIntegerTensor(tensor_a);
    freeIntegerTensor(tensor_b);
    freeIntegerTensor(tensor_c);
}