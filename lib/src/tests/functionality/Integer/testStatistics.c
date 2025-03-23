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
#include "Operations/statistics.h"
#include "Tensor/tensor.h"

#include "Tests/testTensorOperations.h"

void testTensorMean_001() {
    int shape[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    IntegerTensor* t = IntegerTensor_ones(10, shape);
    double mean = IntegerTensor_getMean(t);
    testSuite_assertEquals(1, mean);
}

void testTensorMean_002() {
    int shape[] = {12, 42, 8};
    IntegerTensor* t = IntegerTensor_zeros(3, shape);
    double mean = IntegerTensor_getMean(t);
    testSuite_assertEquals(0, mean);
}

void testTensorStdDev_001() {
    int shape[] = {12, 42, 8};
    IntegerTensor* t = IntegerTensor_zeros(3, shape);
    double stdDev = IntegerTensor_getStandardDeviation(t);
    testSuite_assertEquals(0, stdDev);
}

void testTensorStdDev_002() {
    int shape[] = {3, 2, 2};
    IntegerTensor* t = IntegerTensor_zeros(3, shape);

    for (int i = 0; i < t->base->dataPoints; i++) {
        t->data[i] = i;
    }

    double stdDev = IntegerTensor_getStandardDeviation(t);
    testSuite_assertEquals(3.4520525, stdDev);
}