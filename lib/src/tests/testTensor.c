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

    testTensorAdd_001();
    testTensorDivide_001();
    testTensorMultiply_001();
    testTensorMultiply_002();
    testTensorSubtract_001();
    testTensorConvole1D_001();
    testTensorConvolve2D_001();

    testTensorConvolve3D_001();
    testTensorConvolve3D_002();

    testTensorMSE_001();
    testTensorSAD_001();
    testTensorMAD_001();

    if (ENV_PROFILE_TESTING) {
        profileTensorAdd_001();
        profileTensorDivide_001();
        profileTensorMultiply_001();
        profileTensorSubtract_001();

        profileTensorConvolve3D_001();
    }
}