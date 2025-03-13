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