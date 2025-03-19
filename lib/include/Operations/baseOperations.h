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

#ifndef BASE_OPERATIONS_H
#define BASE_OPERATIONS_H

#include "Tensor/tensor.h"

void checkTensorCompatability(const Tensor* a, const Tensor* b, const char *operation);

void IntegerTensor_multiply(const IntegerTensor* a, const IntegerTensor* b, const IntegerTensor* destination);
void IntegerTensor_divide(const IntegerTensor* a, const IntegerTensor* b, const IntegerTensor* destination);
void IntegerTensor_add(const IntegerTensor* a, const IntegerTensor* b, const IntegerTensor* destination);
void IntegerTensor_subtract(const IntegerTensor* a, const IntegerTensor* b, const IntegerTensor* destination);
void IntegerTensor_scalarMultiply(const IntegerTensor* a, const int scalar, const IntegerTensor* destination);

void FloatTensor_multiply(const FloatTensor* a, const FloatTensor* b, const FloatTensor* destination);
void FloatTensor_divide(const FloatTensor* a, const FloatTensor* b, const FloatTensor* destination);
void FloatTensor_add(const FloatTensor* a, const FloatTensor* b, const FloatTensor* destination);
void FloatTensor_subtract(const FloatTensor* a, const FloatTensor* b, const FloatTensor* destination);
void FloatTensor_scalarMultiply(const FloatTensor* a, const float scalar, const FloatTensor* destination);

void DoubleTensor_multiply(const DoubleTensor* a, const DoubleTensor* b, const DoubleTensor* destination);
void DoubleTensor_divide(const DoubleTensor* a, const DoubleTensor* b, const DoubleTensor* destination);
void DoubleTensor_add(const DoubleTensor* a, const DoubleTensor* b, const DoubleTensor* destination);
void DoubleTensor_subtract(const DoubleTensor* a, const DoubleTensor* b, const DoubleTensor* destination);
void DoubleTensor_scalarMultiply(const DoubleTensor* a, const double scalar, const DoubleTensor* destination);

#endif