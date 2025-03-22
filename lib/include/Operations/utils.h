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

#ifndef UTILS_H
#define UTILS_H

#include "Tensor/tensor.h"

void IntegerTensor_flatten(const IntegerTensor* tensor);
void FloatTensor_flatten(const FloatTensor* tensor);
void DoubleTensor_flatten(const DoubleTensor* tensor);

void IntegerTensor_reshape(const IntegerTensor* tensor, const int* shape, const int dimensions);
void FloatTensor_reshape(const FloatTensor* tensor, const int* shape, const int dimensions);
void DoubleTensor_reshape(const DoubleTensor* tensor, const int* shape, const int dimensions);

#endif