#ifndef TENSOR_OPERATIONS_H
#define TENSOR_OPERATIONS_H

#include "Tensor/tensor.h"

void IntegerTensor_multiply(const IntegerTensor* a, const IntegerTensor* b, const IntegerTensor* destination);
void IntegerTensor_divide(const IntegerTensor* a, const IntegerTensor* b, const IntegerTensor* destination);
void IntegerTensor_add(const IntegerTensor* a, const IntegerTensor* b, const IntegerTensor* destination);
void IntegerTensor_subtract(const IntegerTensor* a, const IntegerTensor* b, const IntegerTensor* destination);
void IntegerTensor_scalarMultiply(const IntegerTensor* a, const int scalar, const IntegerTensor* destination);

#endif