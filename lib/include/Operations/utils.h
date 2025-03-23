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

#include <stdlib.h>
#include <stdint.h>

#include "Tensor/tensor.h"
#include "Operations/compare.h"

void IntegerTensor_flatten(const IntegerTensor* tensor);
void FloatTensor_flatten(const FloatTensor* tensor);
void DoubleTensor_flatten(const DoubleTensor* tensor);

void IntegerTensor_reshape(const IntegerTensor* tensor, const int* shape, const int dimensions);
void FloatTensor_reshape(const FloatTensor* tensor, const int* shape, const int dimensions);
void DoubleTensor_reshape(const DoubleTensor* tensor, const int* shape, const int dimensions);

size_t IntegerTensor_argSearch(const IntegerTensor* tensor, const Integer_SearchFunction searchFunction);
size_t IntegerTensor_argMin(const IntegerTensor* tensor);
size_t IntegerTensor_argMax(const IntegerTensor* tensor);

size_t FloatTensor_argSearch(const FloatTensor* tensor, const Float_SearchFunction searchFunction);
size_t FloatTensor_argMin(const FloatTensor* tensor);
size_t FloatTensor_argMax(const FloatTensor* tensor);

size_t DoubleTensor_argSearch(const DoubleTensor* tensor, const Double_SearchFunction searchFunction);
size_t DoubleTensor_argMin(const DoubleTensor* tensor);
size_t DoubleTensor_argMax(const DoubleTensor* tensor);

void IntegerTensor_clamp(const IntegerTensor* tensor, const IntegerTensor* destination, const int min, const int max);
void FloatTensor_clamp(const FloatTensor* tensor, const FloatTensor* destination, const float min, const float max);
void DoubleTensor_clamp(const DoubleTensor* tensor, const DoubleTensor* destination, const double min, const double max);

#endif