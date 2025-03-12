#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include "testSuite.h"
#include "globals.h"
#include "Tensor/tensor.h"
#include "Operations/tensorOperations.h"
#include "Operations/convolution.h"

#include "Tests/Integer/testTensorOperations.h"

void testIntegerTensor() {
    int shape[4] = {16, 32, 16, 64};
    int dimensions = 4;

    IntegerTensor* tensor = createIntegerTensor(dimensions, shape);

    testSuite_assertEquals(tensor->base->dimensions, dimensions);
    freeIntegerTensor(tensor);
}

void testTensorConvole1D() {
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

    kernel->tensor[0] = 1;
    kernel->tensor[1] = -1;

    IntegerTensor_convolve1D(t, kernel, dest, 1);

    printf("Convolved Matrix: [");

    for (int i = 0; i < shape[0]; i++) {
        printf("%d", dest->tensor[i]);
        
        if (i + 1 < shape[0]) {
            printf(", ");
        }
    }

    printf("]\n");
}

int main() {
    ENV_UNIT_TESTING = 1;

    testTensorAdd_001();
    testTensorDivide_001();
    testTensorMultiply_001();
    testTensorSubtract_001();

    if (ENV_PROFILE_TESTING) {
        profileTensorAdd_001();
        profileTensorDivide_001();
        profileTensorMultiply_001();
        profileTensorSubtract_001();
    }
}