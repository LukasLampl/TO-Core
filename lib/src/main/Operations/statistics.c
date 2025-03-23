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
#include "Operations/statistics.h"
#include "Error/exceptions.h"

/**
 * Gets the sum of a tensor row, starting from the start pointer until the end pointer.
 * 
 * @param *startPtr     Pointer from where to start summing.
 * @param *endPtr       End of summing (exclusive).
 * @param tensorType    Type of tensor to sum up.
 * 
 * @return The sum between the range [*startPtr; *endPtr).
 */
double getSumOfRow(void* startPtr, void* endPtr, const TensorType tensorType) {
    double sum = 0;

    switch (tensorType) {
    case _TENSOR_TYPE_INTEGER_: {
        int* start = (int*)startPtr;
        const int* end = (int*)endPtr;
        
        while (start < end) {
            sum += *start++;
        }
        break;
    }
    case _TENSOR_TYPE_FLOAT_: {
        float* start = (float*)startPtr;
        const float* end = (float*)endPtr;
        
        while (start < end) {
            sum += *start++;
        }
        break;
    }
    case _TENSOR_TYPE_DOUBLE_: {
        double* start = (double*)startPtr;
        const double* end = (double*)endPtr;
        
        while (start < end) {
            sum += *start++;
        }
        break;
    }
    default:
        break;
    }

    return sum;
}

/**
 * Calculates the mean of a given tensor.
 * 
 * @param *tensor       Tensor from which to calculate the mean.
 * @param tensorType    Type of the tensor.
 * 
 * @return The mean of the tensor.
 */
double getMeanValue(const void* tensor, const TensorType tensorType) {
    if (tensor == NULL) {
        (void)throwNullPointerException("Tensor can't be NULL for mean calculation.");
        return 0;
    }

    double mean = 0.0;
    size_t dataPoints = ((Tensor*)getTensorBaseByType(tensor, tensorType))->dataPoints;

    switch (tensorType) {
    case _TENSOR_TYPE_INTEGER_: {
        IntegerTensor* t = (IntegerTensor*)tensor;
        mean = (double)getSumOfRow(t->data, t->data + t->base->dataPoints, _TENSOR_TYPE_INTEGER_);
        break;
    }
    case _TENSOR_TYPE_FLOAT_: {
        FloatTensor* t = (FloatTensor*)tensor;
        mean = (double)getSumOfRow(t->data, t->data + t->base->dataPoints, _TENSOR_TYPE_FLOAT_);
        break;
    }
    case _TENSOR_TYPE_DOUBLE_: {
        DoubleTensor* t = (DoubleTensor*)tensor;
        mean = (double)getSumOfRow(t->data, t->data + t->base->dataPoints, _TENSOR_TYPE_DOUBLE_);
        break;
    }
    default:
        (void)throwIllegalArgumentException("No such tensor type.");
        return 0;
    }

    return mean / (double)dataPoints;
}

/**
 * Calculates the mean of an IntegerTensor.
 * 
 * @param *tensor   Tensor for which to calculate the mean.
 * 
 * @return The mean of the tensor.
 */
double IntegerTensor_getMean(const IntegerTensor* tensor) {
    return (double)getMeanValue(tensor, _TENSOR_TYPE_INTEGER_);
}

/**
 * Calculates the mean of a FloatTensor.
 * 
 * @param *tensor   Tensor for which to calculate the mean.
 * 
 * @return The mean of the tensor.
 */
double FloatTensor_getMean(const FloatTensor* tensor) {
    return (double)getMeanValue(tensor, _TENSOR_TYPE_FLOAT_);
}

/**
 * Calculates the mean of a DoubleTensor.
 * 
 * @param *tensor   Tensor for which to calculate the mean.
 * 
 * @return The mean of the tensor.
 */
double DoubleTensor_getMean(const DoubleTensor* tensor) {
    return (double)getMeanValue(tensor, _TENSOR_TYPE_DOUBLE_);
}

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
        int* start = ((IntegerTensor*)tensor)->data;
        const int* end = start + dataPoints;

        while (start < end) {
            double delta = *start++ - mean;
            sum += delta * delta;
        }
        break;
    }
    case _TENSOR_TYPE_FLOAT_: {
        float* start = ((FloatTensor*)tensor)->data;
        const float* end = start + dataPoints;

        while (start < end) {
            double delta = *start++ - mean;
            sum += delta * delta;
        }
        break;
    }
    case _TENSOR_TYPE_DOUBLE_: {
        double* start = ((DoubleTensor*)tensor)->data;
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