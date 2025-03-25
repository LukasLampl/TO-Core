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
#include "Operations/loss.h"

#include "testSuite.h"

void testTensorMSE_001() {
    printf("TestTensorMSE...\n");
    int shape[] = {5};
    IntegerTensor* t_1 = IntegerTensor_zeros(1, shape);
    IntegerTensor* t_2 = IntegerTensor_zeros(1, shape);

    t_1->data[0] = 0;
    t_1->data[1] = 1;
    t_1->data[2] = 2;
    t_1->data[3] = 3;
    t_1->data[4] = 4;

    t_2->data[0] = 4;
    t_2->data[1] = 3;
    t_2->data[2] = 2;
    t_2->data[3] = 1;
    t_2->data[4] = 0;

    double MSE = IntegerTensor_MSE(t_1, t_2);
    testSuite_assertEquals(8, MSE);
    printf("> Pass\n\n");
}

void testTensorSAD_001() {
    printf("TestTensorSAD_001...\n");
    int shape[] = {5};
    IntegerTensor* t_1 = IntegerTensor_zeros(1, shape);
    IntegerTensor* t_2 = IntegerTensor_zeros(1, shape);

    t_1->data[0] = 0;
    t_1->data[1] = 1;
    t_1->data[2] = 2;
    t_1->data[3] = 3;
    t_1->data[4] = 4;

    t_2->data[0] = 4;
    t_2->data[1] = 3;
    t_2->data[2] = 2;
    t_2->data[3] = 1;
    t_2->data[4] = 0;

    double SAD = IntegerTensor_SAD(t_1, t_2);
    testSuite_assertEquals(12, SAD);
    printf("> Pass\n\n");
}

void testTensorMAD_001() {
    printf("TestTensorMAD_001...\n");
    int shape[] = {5};
    IntegerTensor* t_1 = IntegerTensor_zeros(1, shape);
    IntegerTensor* t_2 = IntegerTensor_zeros(1, shape);

    t_1->data[0] = 0;
    t_1->data[1] = 1;
    t_1->data[2] = 2;
    t_1->data[3] = 3;
    t_1->data[4] = 4;

    t_2->data[0] = 4;
    t_2->data[1] = 3;
    t_2->data[2] = 2;
    t_2->data[3] = 1;
    t_2->data[4] = 0;

    double MAD = IntegerTensor_MAD(t_1, t_2);
    testSuite_assertEquals(2.4, MAD);
    printf("> Pass\n\n");
}