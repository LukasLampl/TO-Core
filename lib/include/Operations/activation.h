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

#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "Tensor/tensor.h"

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

#include <math.h>

#include "Tensor/tensor.h"
#include "Operations/activation.h"
#include "mathUtils.h"
#include "Network/layer.h"

typedef enum {
    RELU,
    LEAKY_RELU,
    SIGMOID,
    TANH
} ActivationType;

typedef struct {
    Layer* base;
    ActivationType type;
    double alpha;
} ActivationLayer;

void IntegerTensor_ReLU(IntegerTensor* tensor);
void FloatTensor_ReLU(FloatTensor* tensor);
void DoubleTensor_ReLU(DoubleTensor* tensor);

void IntegerTensor_LeakyReLU(IntegerTensor* tensor, const int alpha);
void FloatTensor_LeakyReLU(FloatTensor* tensor, const float alpha);
void DoubleTensor_LeakyReLU(DoubleTensor* tensor, const double alpha);

void IntegerTensor_Sigmoid(IntegerTensor* tensor);
void FloatTensor_Sigmoid(FloatTensor* tensor);
void DoubleTensor_Sigmoid(DoubleTensor* tensor);

void IntegerTensor_Tanh(IntegerTensor* tensor);
void FloatTensor_Tanh(FloatTensor* tensor);
void DoubleTensor_Tanh(DoubleTensor* tensor);

ActivationLayer* Integer_createActivationLayer(const ActivationType activationType,
    const int alpha);

ActivationLayer* Float_createActivationLayer(const ActivationType activationType,
    const int alpha);

ActivationLayer* Double_createActivationLayer(const ActivationType activationType,
    const int alpha);

void ActivationLayer_forward(ActivationLayer* layer, void* input);

void ActivationLayer_free(ActivationLayer* layer);

#endif