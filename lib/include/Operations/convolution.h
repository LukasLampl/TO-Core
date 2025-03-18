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

#ifndef CONVOLUTION_H
#define CONVOLUTION_H

#include "Tensor/tensor.h"

void IntegerTensor_convolve(const IntegerTensor* tensor,
    const IntegerTensor* kernel, const IntegerTensor* dest, const int stride);

void FloatTensor_convolve(const FloatTensor* tensor,
    const FloatTensor* kernel, const FloatTensor* dest, const int stride);

void DoubleTensor_convolve(const DoubleTensor* tensor,
    const DoubleTensor* kernel, const DoubleTensor* dest, const int stride);

#endif