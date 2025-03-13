#ifndef CONVOLUTION_H
#define CONVOLUTION_H

#include "Tensor/tensor.h"

void IntegerTensor_convolve1D(const IntegerTensor* tensor,
    const IntegerTensor* kernel, const IntegerTensor* destination,
    const int stride);

void IntegerTensor_convolve2D(const IntegerTensor* tensor,
    const IntegerTensor* kernel, const IntegerTensor* destination,
    const int stride);

#endif