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

#ifndef TENSOR_OPERATIONS_H
#define TENSOR_OPERATIONS_H

#include "Tensor/tensor.h"

void checkTensorCompatability(const IntegerTensor* a, const IntegerTensor* b);

void IntegerTensor_multiply(const IntegerTensor* a, const IntegerTensor* b, const IntegerTensor* destination);
void IntegerTensor_divide(const IntegerTensor* a, const IntegerTensor* b, const IntegerTensor* destination);
void IntegerTensor_add(const IntegerTensor* a, const IntegerTensor* b, const IntegerTensor* destination);
void IntegerTensor_subtract(const IntegerTensor* a, const IntegerTensor* b, const IntegerTensor* destination);
void IntegerTensor_scalarMultiply(const IntegerTensor* a, const int scalar, const IntegerTensor* destination);

#endif