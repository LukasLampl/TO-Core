#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include "testSuite.h"
#include "globals.h"
#include "Tensor/tensor.h"
#include "Operations/tensorOperations.h"
#include "Operations/convolution.h"

#include "Tests/Integer/testTensorOperations.h"

int main() {
    ENV_UNIT_TESTING = 1;

    /*testTensorAdd_001();
    testTensorDivide_001();
    testTensorMultiply_001();
    testTensorSubtract_001();*/
    testTensorConvole1D_001();
    testTensorConvolve2D_001();

    /*if (ENV_PROFILE_TESTING) {
        profileTensorAdd_001();
        profileTensorDivide_001();
        profileTensorMultiply_001();
        profileTensorSubtract_001();
    }*/
}