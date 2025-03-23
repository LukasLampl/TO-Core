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

    t->data[0] = 5;
    t->data[1] = -4;
    t->data[2] = 43;
    t->data[3] = -17;
    t->data[4] = 0;

    kernel->data[0] = 3;
    kernel->data[1] = -17;

    IntegerTensor_convolve(t, kernel, dest, 1);

    printf("\nConvolution 1D\n");
    printf("=======================\n");
    IntegerTensor_print(dest);
    printf("\n");

    testSuite_assertEquals(83, dest->data[0]);
    testSuite_assertEquals(-743, dest->data[1]);
    testSuite_assertEquals(418, dest->data[2]);
    testSuite_assertEquals(-51, dest->data[3]);
}

void testTensorConvole1D_002() {
    int shape[] = {5};
    int kernelShape[] = {2};
    DoubleTensor* t = DoubleTensor_zeros(1, shape);
    DoubleTensor* kernel = DoubleTensor_zeros(1, kernelShape);
    DoubleTensor* dest = DoubleTensor_zeros(1, shape);

    t->data[0] = 5.5;
    t->data[1] = -4.2;
    t->data[2] = 0.0;
    t->data[3] = 67.3;
    t->data[4] = 89.4;

    kernel->data[0] = 3.141;
    kernel->data[1] = -12.6;

    DoubleTensor_convolve(t, kernel, dest, 1);

    printf("\nConvolution 1D\n");
    printf("=======================\n");
    DoubleTensor_print(dest);
    printf("\n");

    testSuite_assertEquals(70.1955, dest->data[0]);
    testSuite_assertEquals(-13.1922, dest->data[1]);
    testSuite_assertEquals(-847.98, dest->data[2]);
    testSuite_assertEquals(-915.0507, dest->data[3]);
}

void testTensorConvolve2D_001() {
    int shape[] = {3, 3};
    int kernelShape[] = {2, 2};
    IntegerTensor* t = IntegerTensor_zeros(2, shape);
    IntegerTensor* kernel = IntegerTensor_zeros(2, kernelShape);
    IntegerTensor* dest = IntegerTensor_zeros(2, shape);

    t->data[0] = 12;      t->data[1] = 5;       t->data[2] = -34;
    t->data[3] = 6;       t->data[4] = 12;      t->data[5] = -4;
    t->data[6] = -7;      t->data[7] = 56;      t->data[8] = 98;

    kernel->data[0] = 6;  kernel->data[1] = -2;
    kernel->data[2] = 3;  kernel->data[3] = -7;

    IntegerTensor_convolve(t, kernel, dest, 1);

    printf("\nConvolution 2D\n");
    printf("=======================\n");
    IntegerTensor_print(dest);
    printf("\n");

    testSuite_assertEquals(-4, dest->data[0]);
    testSuite_assertEquals(162, dest->data[1]);
    testSuite_assertEquals(-401, dest->data[2]);
    testSuite_assertEquals(-438, dest->data[3]);
}

void testTensorConvolve3D_001() {
    int shape[] = {2, 4, 4};
    int kernelShape[] = {2, 3, 3};
    int outputShape[] = {1, 2, 2};
    IntegerTensor* t = IntegerTensor_zeros(3, shape);
    IntegerTensor* kernel = IntegerTensor_zeros(3, kernelShape);
    IntegerTensor* dest = IntegerTensor_zeros(3, outputShape);

    t->data[0] = 12;       t->data[1] = 5;        t->data[2] = -34;      t->data[3] = 67;
    t->data[4] = -45;      t->data[5] = 1;        t->data[6] = -2;       t->data[7] = 7;
    t->data[8] = 34;       t->data[9] = 6;        t->data[10] = -17;     t->data[11] = -2;
    t->data[12] = -36;     t->data[13] = 0;       t->data[14] = 53;      t->data[15] = 0;

    t->data[16] = -6;      t->data[17] = 2;       t->data[18] = -23;     t->data[19] = 34;
    t->data[20] = -76;     t->data[21] = 2;       t->data[22] = 76;      t->data[23] = 1;
    t->data[24] = 75;      t->data[25] = 9;       t->data[26] = -64;     t->data[27] = -5;
    t->data[28] = -28;     t->data[29] = 7;       t->data[30] = 23;      t->data[31] = 2;

    kernel->data[0] = 6;      kernel->data[1] = -2;     kernel->data[2] = 3;
    kernel->data[3] = -1;     kernel->data[4] = 10;     kernel->data[5] = 45;
    kernel->data[6] = 20;     kernel->data[7] = 10;     kernel->data[8] = 12;

    kernel->data[9] = 6;      kernel->data[10] = -2;     kernel->data[11] = 3;
    kernel->data[12] = 30;    kernel->data[13] = -8;     kernel->data[14] = 76;
    kernel->data[15] = 56;    kernel->data[16] = -1;     kernel->data[17] = 32;

    IntegerTensor_convolve(t, kernel, dest, 1);

    printf("\nConvolution 3D\n");
    printf("=======================\n");
    IntegerTensor_print(dest);
    printf("\n");

    testSuite_assertEquals(5975, dest->data[0]);
    testSuite_assertEquals(615, dest->data[1]);
    testSuite_assertEquals(-4858, dest->data[2]);
    testSuite_assertEquals(993, dest->data[3]);
}

void testTensorConvolve3D_002() {
    int shape[] = {2, 4, 5};
    int kernelShape[] = {1, 3, 3};
    int outputShape[] = {2, 2, 3};
    IntegerTensor* t = IntegerTensor_zeros(3, shape);
    IntegerTensor* kernel = IntegerTensor_zeros(3, kernelShape);
    IntegerTensor* dest = IntegerTensor_zeros(3, outputShape);

    t->data[0] = 12;       t->data[1] = 5;        t->data[2] = -34;      t->data[3] = 67;       t->data[4] = -45;
    t->data[5] = 1;        t->data[6] = -2;       t->data[7] = 7;        t->data[8] = 34;       t->data[9] = 6;
    t->data[10] = -17;     t->data[11] = -2;      t->data[12] = -36;     t->data[13] = 0;       t->data[14] = 53;
    t->data[15] = 0;       t->data[16] = -6;      t->data[17] = 2;       t->data[18] = -23;     t->data[19] = 34;

    t->data[20] = -76;     t->data[21] = 2;       t->data[22] = 76;      t->data[23] = 1;       t->data[24] = 75;
    t->data[25] = 9;       t->data[26] = -64;     t->data[27] = -5;      t->data[28] = -28;     t->data[29] = 7;
    t->data[30] = 23;      t->data[31] = 2;       t->data[32] = 23;      t->data[33] = 2;       t->data[34] = -5;
    t->data[35] = 2;       t->data[36] = -7;      t->data[37] = 0;       t->data[38] = 47;      t->data[39] = 89;

    kernel->data[0] = 1;      kernel->data[1] = -2;     kernel->data[2] = 3;
    kernel->data[3] = -1;     kernel->data[4] = 10;     kernel->data[5] = 45;
    kernel->data[6] = 20;     kernel->data[7] = 10;     kernel->data[8] = 12;

    IntegerTensor_convolve(t, kernel, dest, 1);

    printf("\nConvolution 3D\n");
    printf("=======================\n");
    IntegerTensor_print(dest);
    printf("\n");

    testSuite_assertEquals(-598, dest->data[0]);
    testSuite_assertEquals(1476, dest->data[1]);
    testSuite_assertEquals(216, dest->data[2]);
    testSuite_assertEquals(-1633, dest->data[3]);
    testSuite_assertEquals(-648, dest->data[4]);
    testSuite_assertEquals(2596, dest->data[5]);
    testSuite_assertEquals(30, dest->data[6]);
    testSuite_assertEquals(-1099, dest->data[7]);
    testSuite_assertEquals(759, dest->data[8]);
    testSuite_assertEquals(1124, dest->data[9]);
    testSuite_assertEquals(604, dest->data[10]);
    testSuite_assertEquals(1382, dest->data[11]);
}