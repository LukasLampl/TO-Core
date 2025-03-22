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
#include "Operations/mean.h"

/**
 * Calculates the standard deviation of a given tensor.
 * 
 * <p><b>Note:</b><br>
 * Try to avoid using this function and refer to the typed functions instead.
 * </p>
 * 
 * @param *tensor       Tensor for which to calculate the standard deviation.
 * @param tensorType    Type of the given tensor.
 * @param mean          The mean value of the tensor.
 * 
 * @return The calculated standard deviation of the given tensor.
 */
double getStandardDeviation(const void* tensor, const TensorType tensorType, double mean) {
    size_t dataPoints = ((Tensor*)getTensorBaseByType(tensor, tensorType))->dataPoints;
    double sum = 0.0;

    switch (tensorType) {
    case _TENSOR_TYPE_INTEGER_: {
        int* start = ((IntegerTensor*)tensor)->tensor;
        const int* end = start + dataPoints;

        while (start < end) {
            double delta = *start++ - mean;
            sum += delta * delta;
        }
        break;
    }
    case _TENSOR_TYPE_FLOAT_: {
        float* start = ((FloatTensor*)tensor)->tensor;
        const float* end = start + dataPoints;

        while (start < end) {
            double delta = *start++ - mean;
            sum += delta * delta;
        }
        break;
    }
    case _TENSOR_TYPE_DOUBLE_: {
        double* start = ((DoubleTensor*)tensor)->tensor;
        const double* end = start + dataPoints;

        while (start < end) {
            double delta = *start++ - mean;
            sum += delta * delta;
        }
        break;
    }
    }

    return (double)sqrt(sum / (double)dataPoints);
}

/**
 * Calculates the standard deviation of an IntegerTensor.
 * 
 * @param *tensor   Tensor for which to calculate the standard deviation.
 * 
 * @return The standard deviation of the given tensor.
 */
double IntegerTensor_getStandardDeviation(const IntegerTensor* tensor) {
    const double mean = (double)IntegerTensor_getMean(tensor);
   return (double)getStandardDeviation(tensor, _TENSOR_TYPE_INTEGER_, mean);
}

/**
 * Calculates the standard deviation of a FloatTensor.
 * 
 * @param *tensor   Tensor for which to calculate the standard deviation.
 * 
 * @return The standard deviation of the given tensor.
 */
double FloatTensor_getStandardDeviation(const FloatTensor* tensor) {
    const double mean = (double)FloatTensor_getMean(tensor);
    return (double)getStandardDeviation(tensor, _TENSOR_TYPE_FLOAT_, mean);
}

/**
 * Calculates the standard deviation of a DoubleTensor.
 * 
 * @param *tensor   Tensor for which to calculate the standard deviation.
 * 
 * @return The standard deviation of the given tensor.
 */
double DoubleTensor_getStandardDeviation(const DoubleTensor* tensor) {
    const double mean = (double)DoubleTensor_getMean(tensor);
    return (double)getStandardDeviation(tensor, _TENSOR_TYPE_DOUBLE_, mean);
}