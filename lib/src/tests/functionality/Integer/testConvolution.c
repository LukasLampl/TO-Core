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

#include "Tests/Integer/testTensorOperations.h"
#include "Tensor/tensor.h"
#include "Operations/convolution.h"

#include "testSuite.h"

void testTensorConvole1D_001() {
    int shape[] = {5};
    int kernelShape[] = {2};
    IntegerTensor* t = createIntegerTensor(1, shape);
    IntegerTensor* kernel = createIntegerTensor(1, kernelShape);
    IntegerTensor* dest = createIntegerTensor(1, shape);

    t->tensor[0] = 5;
    t->tensor[1] = -4;
    t->tensor[2] = 43;
    t->tensor[3] = -17;
    t->tensor[4] = 0;

    kernel->tensor[0] = 3;
    kernel->tensor[1] = -17;

    IntegerTensor_convolve(t, kernel, dest, 1);

    printf("\nConvolution 1D\n");
    printf("=======================\n");
    IntegerTensor_print(dest);
    printf("\n");
}

void testTensorConvolve2D_001() {
    int shape[] = {3, 3};
    int kernelShape[] = {2, 2};
    IntegerTensor* t = createIntegerTensor(2, shape);
    IntegerTensor* kernel = createIntegerTensor(2, kernelShape);
    IntegerTensor* dest = createIntegerTensor(2, shape);

    t->tensor[0] = 12;      t->tensor[1] = 5;       t->tensor[2] = -34;
    t->tensor[3] = 6;       t->tensor[4] = 12;      t->tensor[5] = -4;
    t->tensor[6] = -7;      t->tensor[7] = 56;      t->tensor[8] = 98;

    kernel->tensor[0] = 6;  kernel->tensor[1] = -2;
    kernel->tensor[2] = 3;  kernel->tensor[3] = -7;

    IntegerTensor_convolve(t, kernel, dest, 1);

    printf("\nConvolution 2D\n");
    printf("=======================\n");
    IntegerTensor_print(dest);
    printf("\n");
}

void testTensorConvolve3D_001() {
    int shape[] = {2, 3, 3};
    int kernelShape[] = {2, 2, 2};
    int outputShape[] = {1, 2, 2};
    IntegerTensor* t = createIntegerTensor(3, shape);
    IntegerTensor* kernel = createIntegerTensor(3, kernelShape);
    IntegerTensor* dest = createIntegerTensor(3, outputShape);

    t->tensor[0] = 12;      t->tensor[1] = 5;       t->tensor[2] = -34;
    t->tensor[3] = 67;      t->tensor[4] = -45;     t->tensor[5] = 1;
    t->tensor[6] = -2;      t->tensor[7] = 7;       t->tensor[8] = 0;

    t->tensor[9] = 3;       t->tensor[10] = 78;     t->tensor[11] = 0;
    t->tensor[12] = 2;      t->tensor[13] = 65;     t->tensor[14] = 13;
    t->tensor[15] = 1;      t->tensor[16] = 7;      t->tensor[17] = 33;

    kernel->tensor[0] = 6;  kernel->tensor[1] = -2;
    kernel->tensor[2] = 3;  kernel->tensor[3] = -1;

    kernel->tensor[4] = 4;  kernel->tensor[5] = -12;
    kernel->tensor[6] = 9;  kernel->tensor[7] = -18;

    IntegerTensor_convolve(t, kernel, dest, 1);

    printf("\nConvolution 3D\n");
    printf("=======================\n");
    IntegerTensor_print(dest);
    printf("\n");
}