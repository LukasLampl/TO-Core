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

#include "Tests/testTensorOperations.h"
#include "Tensor/tensor.h"
#include "Operations/convolution.h"

#include "testSuite.h"

void testTensorConvole1D_001() {
    int shape[] = {5};
    int kernelShape[] = {2};
    IntegerTensor* t = IntegerTensor_zeros(1, shape);
    IntegerTensor* kernel = IntegerTensor_zeros(1, kernelShape);
    IntegerTensor* dest = IntegerTensor_zeros(1, shape);

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

    testSuite_assertEquals(83, dest->tensor[0]);
    testSuite_assertEquals(-743, dest->tensor[1]);
    testSuite_assertEquals(418, dest->tensor[2]);
    testSuite_assertEquals(-51, dest->tensor[3]);
}

void testTensorConvole1D_002() {
    int shape[] = {5};
    int kernelShape[] = {2};
    DoubleTensor* t = DoubleTensor_zeros(1, shape);
    DoubleTensor* kernel = DoubleTensor_zeros(1, kernelShape);
    DoubleTensor* dest = DoubleTensor_zeros(1, shape);

    t->tensor[0] = 5.5;
    t->tensor[1] = -4.2;
    t->tensor[2] = 0.0;
    t->tensor[3] = 67.3;
    t->tensor[4] = 89.4;

    kernel->tensor[0] = 3.141;
    kernel->tensor[1] = -12.6;

    DoubleTensor_convolve(t, kernel, dest, 1);

    printf("\nConvolution 1D\n");
    printf("=======================\n");
    DoubleTensor_print(dest);
    printf("\n");

    testSuite_assertEquals(70.1955, dest->tensor[0]);
    testSuite_assertEquals(-13.1922, dest->tensor[1]);
    testSuite_assertEquals(-847.98, dest->tensor[2]);
    testSuite_assertEquals(-915.0507, dest->tensor[3]);
}

void testTensorConvolve2D_001() {
    int shape[] = {3, 3};
    int kernelShape[] = {2, 2};
    IntegerTensor* t = IntegerTensor_zeros(2, shape);
    IntegerTensor* kernel = IntegerTensor_zeros(2, kernelShape);
    IntegerTensor* dest = IntegerTensor_zeros(2, shape);

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

    testSuite_assertEquals(-4, dest->tensor[0]);
    testSuite_assertEquals(162, dest->tensor[1]);
    testSuite_assertEquals(-401, dest->tensor[2]);
    testSuite_assertEquals(-438, dest->tensor[3]);
}

void testTensorConvolve3D_001() {
    int shape[] = {2, 4, 4};
    int kernelShape[] = {2, 3, 3};
    int outputShape[] = {1, 2, 2};
    IntegerTensor* t = IntegerTensor_zeros(3, shape);
    IntegerTensor* kernel = IntegerTensor_zeros(3, kernelShape);
    IntegerTensor* dest = IntegerTensor_zeros(3, outputShape);

    t->tensor[0] = 12;       t->tensor[1] = 5;        t->tensor[2] = -34;      t->tensor[3] = 67;
    t->tensor[4] = -45;      t->tensor[5] = 1;        t->tensor[6] = -2;       t->tensor[7] = 7;
    t->tensor[8] = 34;       t->tensor[9] = 6;        t->tensor[10] = -17;     t->tensor[11] = -2;
    t->tensor[12] = -36;     t->tensor[13] = 0;       t->tensor[14] = 53;      t->tensor[15] = 0;

    t->tensor[16] = -6;      t->tensor[17] = 2;       t->tensor[18] = -23;     t->tensor[19] = 34;
    t->tensor[20] = -76;     t->tensor[21] = 2;       t->tensor[22] = 76;      t->tensor[23] = 1;
    t->tensor[24] = 75;      t->tensor[25] = 9;       t->tensor[26] = -64;     t->tensor[27] = -5;
    t->tensor[28] = -28;     t->tensor[29] = 7;       t->tensor[30] = 23;      t->tensor[31] = 2;

    kernel->tensor[0] = 6;      kernel->tensor[1] = -2;     kernel->tensor[2] = 3;
    kernel->tensor[3] = -1;     kernel->tensor[4] = 10;     kernel->tensor[5] = 45;
    kernel->tensor[6] = 20;     kernel->tensor[7] = 10;     kernel->tensor[8] = 12;

    kernel->tensor[9] = 6;      kernel->tensor[10] = -2;     kernel->tensor[11] = 3;
    kernel->tensor[12] = 30;    kernel->tensor[13] = -8;     kernel->tensor[14] = 76;
    kernel->tensor[15] = 56;    kernel->tensor[16] = -1;     kernel->tensor[17] = 32;

    IntegerTensor_convolve(t, kernel, dest, 1);

    printf("\nConvolution 3D\n");
    printf("=======================\n");
    IntegerTensor_print(dest);
    printf("\n");

    testSuite_assertEquals(5975, dest->tensor[0]);
    testSuite_assertEquals(615, dest->tensor[1]);
    testSuite_assertEquals(-4858, dest->tensor[2]);
    testSuite_assertEquals(993, dest->tensor[3]);
}

void testTensorConvolve3D_002() {
    int shape[] = {2, 4, 5};
    int kernelShape[] = {1, 3, 3};
    int outputShape[] = {2, 2, 3};
    IntegerTensor* t = IntegerTensor_zeros(3, shape);
    IntegerTensor* kernel = IntegerTensor_zeros(3, kernelShape);
    IntegerTensor* dest = IntegerTensor_zeros(3, outputShape);

    t->tensor[0] = 12;       t->tensor[1] = 5;        t->tensor[2] = -34;      t->tensor[3] = 67;       t->tensor[4] = -45;
    t->tensor[5] = 1;        t->tensor[6] = -2;       t->tensor[7] = 7;        t->tensor[8] = 34;       t->tensor[9] = 6;
    t->tensor[10] = -17;     t->tensor[11] = -2;      t->tensor[12] = -36;     t->tensor[13] = 0;       t->tensor[14] = 53;
    t->tensor[15] = 0;       t->tensor[16] = -6;      t->tensor[17] = 2;       t->tensor[18] = -23;     t->tensor[19] = 34;

    t->tensor[20] = -76;     t->tensor[21] = 2;       t->tensor[22] = 76;      t->tensor[23] = 1;       t->tensor[24] = 75;
    t->tensor[25] = 9;       t->tensor[26] = -64;     t->tensor[27] = -5;      t->tensor[28] = -28;     t->tensor[29] = 7;
    t->tensor[30] = 23;      t->tensor[31] = 2;       t->tensor[32] = 23;      t->tensor[33] = 2;       t->tensor[34] = -5;
    t->tensor[35] = 2;       t->tensor[36] = -7;      t->tensor[37] = 0;       t->tensor[38] = 47;      t->tensor[39] = 89;

    kernel->tensor[0] = 1;      kernel->tensor[1] = -2;     kernel->tensor[2] = 3;
    kernel->tensor[3] = -1;     kernel->tensor[4] = 10;     kernel->tensor[5] = 45;
    kernel->tensor[6] = 20;     kernel->tensor[7] = 10;     kernel->tensor[8] = 12;

    IntegerTensor_convolve(t, kernel, dest, 1);

    printf("\nConvolution 3D\n");
    printf("=======================\n");
    IntegerTensor_print(dest);
    printf("\n");

    testSuite_assertEquals(-598, dest->tensor[0]);
    testSuite_assertEquals(1476, dest->tensor[1]);
    testSuite_assertEquals(216, dest->tensor[2]);
    testSuite_assertEquals(-1633, dest->tensor[3]);
    testSuite_assertEquals(-648, dest->tensor[4]);
    testSuite_assertEquals(2596, dest->tensor[5]);
    testSuite_assertEquals(30, dest->tensor[6]);
    testSuite_assertEquals(-1099, dest->tensor[7]);
    testSuite_assertEquals(759, dest->tensor[8]);
    testSuite_assertEquals(1124, dest->tensor[9]);
    testSuite_assertEquals(604, dest->tensor[10]);
    testSuite_assertEquals(1382, dest->tensor[11]);
}