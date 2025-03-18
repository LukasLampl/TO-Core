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

#include "Tests/Integer/testTensorOperations.h"
#include "Tensor/tensor.h"
#include "Operations/convolution.h"

#include "testSuite.h"

void profileTensorConvolve3D_001() {
    int shape[] = {3, 1920, 1080};
    int kernelShape[] = {3, 3, 3};
    int outputShape[] = {1, 1920, 1080};
    IntegerTensor* t = createIntegerTensor(3, shape);
    IntegerTensor* kernel = createIntegerTensor(3, kernelShape);
    IntegerTensor* dest = createIntegerTensor(3, outputShape);

    printf("\nPreparing conolution 3D of %ld elements.\n", t->base->dataPoints);

    for (int i = 0; i < t->base->dataPoints; i++) {
        t->tensor[i] = 1;
    }

    for (int i = 0; i < kernel->base->dataPoints; i++) {
        kernel->tensor[i] = 1;
    }

    clock_t start = clock();
    IntegerTensor_convolve(t, kernel, dest, 3);
    clock_t end = clock();

    printf(" > Time: %f seconds\n", ((double)(end - start) / (double)CLOCKS_PER_SEC));

    size_t sum = 0;

    for (int i = 0; i < dest->base->dataPoints; i++) {
        sum += dest->tensor[i];
    }

    testSuite_assertEquals(shape[0] * shape[1] * shape[2], sum);
    printf(" > Sum: %ld\n", sum);
}