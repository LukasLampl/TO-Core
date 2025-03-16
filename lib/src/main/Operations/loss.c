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

#include <stdlib.h>

#include "Operations/tensorOperations.h"
#include "Tensor/tensor.h"
#include "Error/exceptions.h"
#include "math.h"

/**
 * Calculates the MSE (= Mean Square Error) between two given tensors.
 * 
 * <p><b>Note:</b><br>
 * Both Tensors must have the same dimensions and shape.
 * </p>
 * 
 * @param *a    Tensor a to check.
 * @param *b    Tensor b to check against.
 * 
 * @return The MSE between both tensors.
 */
double IntegerTensor_MSE(const IntegerTensor* a, const IntegerTensor *b) {
    (void)checkTensorCompatability(a, b);

    double MSE = 0.0;
    const int* data_a = a->tensor;
    const int* data_b = b->tensor;
    const int* endPtr = data_a + a->base->dataPoints;

    while (data_a < endPtr) {
        const int delta = (int)int_abs((*data_a++) - (*data_b++));
        MSE += delta * delta;
    }

    return MSE / a->base->dataPoints;
}

/**
 * Calculates the SAD (= Sum Absolute Difference) between two given tensors.
 * 
 * <p><b>Note:</b><br>
 * Both Tensors must have the same dimensions and shape.
 * </p>
 * 
 * @param *a    Tensor a to check.
 * @param *b    Tensor b to check against.
 * 
 * @return The SAD between both tensors.
 */
double IntegerTensor_SAD(const IntegerTensor* a, const IntegerTensor *b) {
    (void)checkTensorCompatability(a, b);

    double SAD = 0.0;
    const int* data_a = a->tensor;
    const int* data_b = b->tensor;
    const int* endPtr = data_a + a->base->dataPoints;

    while (data_a < endPtr) {
        const int delta = (int)int_abs((*data_a++) - (*data_b++));
        SAD += delta;
    }

    return SAD;
}

/**
 * Calculates the MAD (= Mean Absolute Difference) between two given tensors.
 * 
 * <p><b>Note:</b><br>
 * Both Tensors must have the same dimensions and shape.
 * </p>
 * 
 * @param *a    Tensor a to check.
 * @param *b    Tensor b to check against.
 * 
 * @return The MAD between both tensors.
 */
double IntegerTensor_MAD(const IntegerTensor* a, const IntegerTensor *b) {
    double SAD = IntegerTensor_SAD(a, b);
    return SAD / a->base->dataPoints;
}

/**
 * Calculates the Huber Loss between two given tensors.
 * 
 * <p><b>Note:</b><br>
 * Both Tensors must have the same dimensions and shape.
 * </p>
 * 
 * @param *a    Tensor a to check.
 * @param *b    Tensor b to check against.
 * 
 * @return The Huber Loss between both tensors.
 */
double IntegerTensor_Huber_Loss(const IntegerTensor* a, const IntegerTensor *b, const int delta) {
    (void)checkTensorCompatability(a, b);

    double loss = 0.0;
    const int halfDelta = delta >> 1;
    const int* data_a = a->tensor;
    const int* data_b = b->tensor;
    const int* endPtr = data_a + a->base->dataPoints;

    while (data_a < endPtr) {
        const int a = (int)int_abs((*data_a++) - (*data_b++));
        
        if (a <= delta) {
            loss += (a * a) / 2;
        } else {
            loss += delta * (a - halfDelta);
        }
    }

    return loss;
}