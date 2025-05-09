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

#include "testSuite.h"
#include "Operations/utils.h"
#include "Tensor/tensor.h"

#include "Tests/testTensorOperations.h"

void testTensorArgMin_001() {
    printf("TestTensorArgMin_001...\n");
    int shape[] = {2, 3};
    IntegerTensor* tensor = IntegerTensor_zeros(2, shape);

    for (int i = 0; i < tensor->base->dataPoints; i++) {
        tensor->data[i] = tensor->base->dataPoints - i;
    }

    size_t minIndex = IntegerTensor_argMin(tensor);
    printf("Min index: %ld\n", minIndex);

    testSuite_assertEquals(tensor->base->dataPoints - 1, minIndex);
    printf("> Pass\n\n");
}

void testTensorArgMax_001() {
    printf("TestTensorArgMax_001...\n");
    int shape[] = {2, 3};
    IntegerTensor* tensor = IntegerTensor_zeros(2, shape);

    for (int i = 0; i < tensor->base->dataPoints; i++) {
        tensor->data[i] = tensor->base->dataPoints - i;
    }

    size_t minIndex = IntegerTensor_argMax(tensor);
    printf("Max index: %ld\n", minIndex);

    testSuite_assertEquals(0, minIndex);
    printf("> Pass\n\n");
}

void testTensorClamp_001() {
    printf("TestTensorClamp_001...\n");
    int shape[] = {2, 3};
    int max = 4;
    int min = 0;
    IntegerTensor* tensor = IntegerTensor_zeros(2, shape);
    IntegerTensor* dest = IntegerTensor_zeros(2, shape);

    for (int i = 0; i < tensor->base->dataPoints; i++) {
        tensor->data[i] = i;
    }

    IntegerTensor_clamp(tensor, dest, min, max);
    IntegerTensor_print(dest);

    for (int i = 0; i < tensor->base->dataPoints; i++) {
        const int num = dest->data[i];
        (void)testSuite_assertInBetween(num, min, max);
    }

    printf("> Pass\n\n");
}

void testTensorClamp_002() {
    printf("TestTensorClamp_002...\n");
    int shape[] = {2, 3};
    float min = 0.2;
    float max = 4.75;
    FloatTensor* tensor = FloatTensor_zeros(2, shape);
    FloatTensor* dest = FloatTensor_zeros(2, shape);

    for (int i = 0; i < tensor->base->dataPoints; i++) {
        tensor->data[i] = i * 3.141;
    }

    FloatTensor_clamp(tensor, dest, min, max);
    FloatTensor_print(dest);

    for (int i = 0; i < tensor->base->dataPoints; i++) {
        const float num = dest->data[i];
        (void)testSuite_assertInBetween(num, min, max);
    }

    printf("> Pass\n\n");
}

void testTensorClamp_003() {
    printf("TestTensorClamp_003...\n");
    int shape[] = {2, 3};
    double min = 0.4125645;
    double max = 5.2139874;
    DoubleTensor* tensor = DoubleTensor_zeros(2, shape);
    DoubleTensor* dest = DoubleTensor_zeros(2, shape);

    for (int i = 0; i < tensor->base->dataPoints; i++) {
        tensor->data[i] = i * 3.14159265358979;
    }

    DoubleTensor_clamp(tensor, dest, min, max);
    DoubleTensor_print(dest);

    for (int i = 0; i < tensor->base->dataPoints; i++) {
        const double num = dest->data[i];
        (void)testSuite_assertInBetween(num, min, max);
    }

    printf("> Pass\n\n");
}