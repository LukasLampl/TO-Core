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
#include "Operations/tensorOperations.h"

void testTensorMultiply_001();
void testTensorAdd_001();
void testTensorDivide_001();
void testTensorSubtract_001();

void profileTensorMultiply_001();
void profileTensorAdd_001();
void profileTensorDivide_001();
void profileTensorSubtract_001();

void testTensorConvole1D_001();
void testTensorConvolve2D_001();

#endif