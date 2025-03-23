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

#include <string.h>
#include <limits.h>
#include <stdint.h>

#include "Tensor/tensor.h"
#include "Operations/baseOperations.h"
#include "Operations/utils.h"
#include "Operations/compare.h"
#include "Error/exceptions.h"

#define true 1
#define false 0

/**
 * Flattens a given tensor, by resolving the shape and
 * setting the dimension to `1`. This can be used, since all
 * the data is already stored in a 1D-Array.
 * 
 * @param *tensor   Base of a tensor to flatten.
 */
void flatten(Tensor* tensor) {
    int* shape = (int*)realloc(tensor->shape, sizeof(int));

    if (shape == NULL) {
        (void)throwMemoryAllocationException("While trying to allocate memory for tensor flattening.");
        return;
    }

    shape[0] = tensor->dataPoints;
    tensor->shape = shape;
    tensor->dimensions = 1;
}

/**
 * Flattens a given tensor from N-Dimensions to 1-Dimension.
 * 
 * @param *tensor   Tensor to flatten.
 */
void IntegerTensor_flatten(const IntegerTensor* tensor) {
    (void)flatten(tensor->base);
}

/**
 * Flattens a given tensor from N-Dimensions to 1-Dimension.
 * 
 * @param *tensor   Tensor to flatten.
 */
void FloatTensor_flatten(const FloatTensor* tensor) {
    (void)flatten(tensor->base);
}

/**
 * Flattens a given tensor from N-Dimensions to 1-Dimension.
 * 
 * @param *tensor   Tensor to flatten.
 */
void DoubleTensor_flatten(const DoubleTensor* tensor) {
    (void)flatten(tensor->base);
}

/**
 * Reshapes a given tensor to the new shape and dimensions.
 * 
 * The new shape must match with the old number of datapoints, or else
 * an exception will be thrown.
 * 
 * @param *tensor       Tensor base to reshape.
 * @param *newShape     New shape of the tensor.
 * @param dimensions    The number of dimensions of the new shape.
 * 
 * @throws IllegalArgumentException - When the new number of datapoints does not match the old one.
 */
void reshape(Tensor* tensor, const int* newShape, const int dimensions) {
        size_t newDataPoints = (size_t)countNumberOfDataIndexes(dimensions, newShape);

        if (newDataPoints != tensor->dataPoints) {
            (void)throwIllegalArgumentException("Reshaping must result in equal number of datapoints.");
            return;
        }

        int* shape_copy = (int*)malloc(dimensions * sizeof(int));

        if (shape_copy == NULL) {
            (void)throwMemoryAllocationException("An error occured while trying to reshape a tensor.");
        }
    
        (void)memcpy(shape_copy, newShape, dimensions * sizeof(int));
        (void)free(tensor->shape);
        tensor->shape = shape_copy;
        tensor->dimensions = dimensions;
}

/**
 * Reshapes a given tensor to the given shape.
 * 
 * @param *tensor       Tensor to reshape.
 * @param *shape        The new shape.
 * @param dimensions    Number of dimensions in the new shape.
 * 
 * @throws IllegalArgumentException - When the number of datapoints of the new shape does not the old one.
 */
void IntegerTensor_reshape(const IntegerTensor* tensor, const int* shape, const int dimensions) {
    (void)reshape(tensor->base, shape, dimensions);
}

/**
 * Reshapes a given tensor to the given shape.
 * 
 * @param *tensor       Tensor to reshape.
 * @param *shape        The new shape.
 * @param dimensions    Number of dimensions in the new shape.
 * 
 * @throws IllegalArgumentException - When the number of datapoints of the new shape does not the old one.
 */
void FloatTensor_reshape(const FloatTensor* tensor, const int* shape, const int dimensions) {
    (void)reshape(tensor->base, shape, dimensions);
}

/**
 * Reshapes a given tensor to the given shape.
 * 
 * @param *tensor       Tensor to reshape.
 * @param *shape        The new shape.
 * @param dimensions    Number of dimensions in the new shape.
 * 
 * @throws IllegalArgumentException - When the number of datapoints of the new shape does not the old one.
 */
void DoubleTensor_reshape(const DoubleTensor* tensor, const int* shape, const int dimensions) {
    (void)reshape(tensor->base, shape, dimensions);
}

/**
 * Executes a search over the whole tensor using the given search function and
 * returns the index, at which the search function had its peak.
 * 
 * <p><b>Note:</b><br>
 * If you wan to add a new SearchFunction, please refer to "Operations/compare.h".
 * Just remember, that the output of the SearchFunction must always be `true` or `1`,
 * when the searching operator is valid. For instance: `Integer_isMin(int, int)`
 * returns only `true`, when the first arg is smaller than the second.
 * </p>
 * 
 * @param *tensor           Tensor in which to search.
 * @param searchFunction    The search function to apply.
 * 
 * @return
 * <ul>
 * <li>The index at which the search function had its peak.
 * <li>`SIZE_MAX` when an error occured.
 * </ul>
 */
size_t IntegerTensor_argSearch(const IntegerTensor* tensor, const Integer_SearchFunction searchFunction) {
    if (tensor == NULL) {
        (void)throwNullPointerException("Tensor must not be NULL.");
        return SIZE_MAX;
    }

    size_t index = 0;
    size_t currentIndex = 0;
    int peak = (int)tensor->data[0];

    int* start = tensor->data;
    const int* end = start + tensor->base->dataPoints;

    while (start < end) {
        const int num = *start;

        if ((int)searchFunction(num, peak) == true) {
            peak = num;
            index = currentIndex;
        }

        start++;
        currentIndex++;
    }

    return index;
}

/**
 * Executes a search over the whole tensor using the given search function and
 * returns the index, at which the search function had its peak.
 * 
 * <p><b>Note:</b><br>
 * If you wan to add a new SearchFunction, please refer to "Operations/compare.h".
 * Just remember, that the output of the SearchFunction must always be `true` or `1`,
 * when the searching operator is valid. For instance: `Integer_isMin(int, int)`
 * returns only `true`, when the first arg is smaller than the second.
 * </p>
 * 
 * @param *tensor           Tensor in which to search.
 * @param searchFunction    The search function to apply.
 * 
 * @return
 * <ul>
 * <li>The index at which the search function had its peak.
 * <li>`SIZE_MAX` when an error occured.
 * </ul>
 */
size_t FloatTensor_argSearch(const FloatTensor* tensor, const Float_SearchFunction searchFunction) {
    if (tensor == NULL) {
        (void)throwNullPointerException("Tensor must not be NULL.");
        return SIZE_MAX;
    }

    size_t index = 0;
    size_t currentIndex = 0;
    float peak = (float)tensor->data[0];

    float* start = tensor->data;
    const float* end = start + tensor->base->dataPoints;

    while (start < end) {
        const float num = *start;

        if ((float)searchFunction(num, peak) == true) {
            peak = num;
            index = currentIndex;
        }

        start++;
        currentIndex++;
    }

    return index;
}

/**
 * Executes a search over the whole tensor using the given search function and
 * returns the index, at which the search function had its peak.
 * 
 * <p><b>Note:</b><br>
 * If you wan to add a new SearchFunction, please refer to "Operations/compare.h".
 * Just remember, that the output of the SearchFunction must always be `true` or `1`,
 * when the searching operator is valid. For instance: `Integer_isMin(int, int)`
 * returns only `true`, when the first arg is smaller than the second.
 * </p>
 * 
 * @param *tensor           Tensor in which to search.
 * @param searchFunction    The search function to apply.
 * 
 * @return
 * <ul>
 * <li>The index at which the search function had its peak.
 * <li>`SIZE_MAX` when an error occured.
 * </ul>
 */
size_t DoubleTensor_argSearch(const DoubleTensor* tensor, const Double_SearchFunction searchFunction) {
    if (tensor == NULL) {
        (void)throwNullPointerException("Tensor must not be NULL.");
        return SIZE_MAX;
    }

    size_t index = 0;
    size_t currentIndex = 0;
    double peak = (double)tensor->data[0];

    double* start = tensor->data;
    const double* end = start + tensor->base->dataPoints;

    while (start < end) {
        const double num = *start;

        if ((double)searchFunction(num, peak) == true) {
            peak = num;
            index = currentIndex;
        }

        start++;
        currentIndex++;
    }

    return index;
}

/**
 * Returns the index where the tensor's smallest value lies.
 * 
 * @param *tensor   Tensor in which to search.
 * 
 * @return
 * <ul>
 * <li>The index of the smallest value in the tensor.
 * <li>`SIZE_MAX` when an error occured.
 * </ul>
 */
size_t IntegerTensor_argMin(const IntegerTensor* tensor) {
    return (size_t)IntegerTensor_argSearch(tensor, Integer_isMin);
}

/**
 * Returns the index where the tensor's greatest value lies.
 * 
 * @param *tensor   Tensor in which to search.
 * 
 * @return
 * <ul>
 * <li>The index of the greatest value in the tensor.
 * <li>`SIZE_MAX` when an error occured.
 * </ul>
 */
size_t IntegerTensor_argMax(const IntegerTensor* tensor) {
    return (size_t)IntegerTensor_argSearch(tensor, Integer_isMax);
}

/**
 * Returns the index where the tensor's smallest value lies.
 * 
 * @param *tensor   Tensor in which to search.
 * 
 * @return
 * <ul>
 * <li>The index of the smallest value in the tensor.
 * <li>`SIZE_MAX` when an error occured.
 * </ul>
 */
size_t FloatTensor_argMin(const FloatTensor* tensor) {
    return (size_t)FloatTensor_argSearch(tensor, Float_isMin);
}

/**
 * Returns the index where the tensor's greatest value lies.
 * 
 * @param *tensor   Tensor in which to search.
 * 
 * @return
 * <ul>
 * <li>The index of the greatest value in the tensor.
 * <li>`SIZE_MAX` when an error occured.
 * </ul>
 */
size_t FloatTensor_argMax(const FloatTensor* tensor) {
    return (size_t)FloatTensor_argSearch(tensor, Float_isMax);
}

/**
 * Returns the index where the tensor's smallest value lies.
 * 
 * @param *tensor   Tensor in which to search.
 * 
 * @return
 * <ul>
 * <li>The index of the smallest value in the tensor.
 * <li>`SIZE_MAX` when an error occured.
 * </ul>
 */
size_t DoubleTensor_argMin(const DoubleTensor* tensor) {
    return (size_t)DoubleTensor_argSearch(tensor, Double_isMin);
}

/**
 * Returns the index where the tensor's greatest value lies.
 * 
 * @param *tensor   Tensor in which to search.
 * 
 * @return
 * <ul>
 * <li>The index of the greatest value in the tensor.
 * <li>`SIZE_MAX` when an error occured.
 * </ul>
 */
size_t DoubleTensor_argMax(const DoubleTensor* tensor) {
    return (size_t)DoubleTensor_argSearch(tensor, Double_isMax);
}

/**
 * Clamps a given value.
 * 
 * @param value     The value to clamp.
 * @param min       Minimum the value should have.
 * @param max       Maximum the value should have.
 * 
 * @return
 * <ul>
 * <li>`value` - When `min` < `value` < `max`.
 * <li>`min` - When `min` >= `value`.
 * <li>`max` - When `max` <= `value`.
 * </ul>
 */
int Integer_clamp(const int value, const int min, const int max) {
    return value >= max ? max : value <= min ? min : value;
}

/**
 * Clamps a given value.
 * 
 * @param value     The value to clamp.
 * @param min       Minimum the value should have.
 * @param max       Maximum the value should have.
 * 
 * @return
 * <ul>
 * <li>`value` - When `min` < `value` < `max`.
 * <li>`min` - When `min` >= `value`.
 * <li>`max` - When `max` <= `value`.
 * </ul>
 */
float Float_clamp(const float value, const float min, const float max) {
    return value >= max ? max : value <= min ? min : value;
}

/**
 * Clamps a given value.
 * 
 * @param value     The value to clamp.
 * @param min       Minimum the value should have.
 * @param max       Maximum the value should have.
 * 
 * @return
 * <ul>
 * <li>`value` - When `min` < `value` < `max`.
 * <li>`min` - When `min` >= `value`.
 * <li>`max` - When `max` <= `value`.
 * </ul>
 */
double Double_clamp(const double value, const double min, const double max) {
    return value >= max ? max : value <= min ? min : value;
}

/**
 * Clamps all values in the given tensor and sets the results into the given destination
 * tensor.
 * 
 * @param *tensor       Tensor which to clamp.
 * @param *destination  Destination tensor in which to write the clamped values.
 * @param min           The minimum allowed value.
 * @param max           The maximum allowed value.
 * 
 * @throws IllegalArgumentException - When the given tensor and destination do not match in shape.
 */
void IntegerTensor_clamp(const IntegerTensor* tensor, const IntegerTensor* destination,
    const int min, const int max) {
    (void)checkTensorCompatability(tensor->base, destination->base, "clamping");

    int* dest = destination->data;
    int* start = tensor->data;
    const int* end = start + tensor->base->dataPoints;

    while (start < end) {
        *dest++ = (int)Integer_clamp(*start++, min, max);
    }
}

/**
 * Clamps all values in the given tensor and sets the results into the given destination
 * tensor.
 * 
 * @param *tensor       Tensor which to clamp.
 * @param *destination  Destination tensor in which to write the clamped values.
 * @param min           The minimum allowed value.
 * @param max           The maximum allowed value.
 * 
 * @throws IllegalArgumentException - When the given tensor and destination do not match in shape.
 */
void FloatTensor_clamp(const FloatTensor* tensor, const FloatTensor* destination,
    const float min, const float max) {
    (void)checkTensorCompatability(tensor->base, destination->base, "clamping");

    float* dest = destination->data;
    float* start = tensor->data;
    const float* end = start + tensor->base->dataPoints;

    while (start < end) {
        *dest++ = (float)Float_clamp(*start++, min, max);
    }
}

/**
 * Clamps all values in the given tensor and sets the results into the given destination
 * tensor.
 * 
 * @param *tensor       Tensor which to clamp.
 * @param *destination  Destination tensor in which to write the clamped values.
 * @param min           The minimum allowed value.
 * @param max           The maximum allowed value.
 * 
 * @throws IllegalArgumentException - When the given tensor and destination do not match in shape.
 */
void DoubleTensor_clamp(const DoubleTensor* tensor, const DoubleTensor* destination,
    const double min, const double max) {
    (void)checkTensorCompatability(tensor->base, destination->base, "clamping");

    double* dest = destination->data;
    double* start = tensor->data;
    const double* end = start + tensor->base->dataPoints;

    while (start < end) {
        *dest++ = (double)Double_clamp(*start++, min, max);
    }
}