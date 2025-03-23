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

#ifndef TEST_INTEGER_TENSOR_OPERATIONS_H
#define TEST_INTEGER_TENSOR_OPERATIONS_H

#include <stdio.h>
#include <stdlib.h>

#include "Tensor/tensor.h"
#include "Operations/baseOperations.h"

void testTensorMultiply_001();
void testTensorMultiply_002();
void testTensorAdd_001();
void testTensorDivide_001();
void testTensorSubtract_001();

void testTensorMean_001();
void testTensorMean_002();

void testTensorStdDev_001();
void testTensorStdDev_002();

void profileTensorMultiply_001();
void profileTensorAdd_001();
void profileTensorDivide_001();
void profileTensorSubtract_001();



void testTensorConvole1D_001();
void testTensorConvole1D_002();
void testTensorConvolve2D_001();
void testTensorConvolve3D_001();
void testTensorConvolve3D_002();

void profileTensorConvolve3D_001();



void testTensorMSE_001();
void testTensorSAD_001();
void testTensorMAD_001();



void testTensorArgMin_001();

void testTensorArgMax_001();

void testTensorClamp_001();
void testTensorClamp_002();
void testTensorClamp_003();

#endif