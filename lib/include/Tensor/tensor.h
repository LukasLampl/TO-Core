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

#ifndef TENSOR_H
#define TENSOR_H

#include <stdlib.h>

typedef enum TensorType {
    _TENSOR_TYPE_INTEGER_,
    _TENSOR_TYPE_FLOAT_,
    _TENSOR_TYPE_DOUBLE_
} TensorType;

/**
 * The base of all tensor subtypes with the datatypes
 * all tensors need to identify. It contains the following
 * fields:
 * 
 * <ul>
 * <li>Dimensions of the tensor.</li>
 * <li>Shape of each dimensions.</li>
 * <li>Number of elements / datapoints in the tensor.</li>
 * </ul>
 */
typedef struct {
    /**
     * Number of dimensions the tensor consists of.
     */
    int dimensions;
 
    /**
     * Array of the sizes of each dimension.
     */
    int* shape;

    /**
     * Number of datapoints / elements in the tensor.
     */
    size_t dataPoints;
} Tensor;

/**
 * A Tensor with the type of the data as INTEGER.
 */
typedef struct {
    /**
     * Metadata of the tensor.
     */
    Tensor* base;
 
    /**
     * Data of the tensor.
     */
    int *tensor;
} IntegerTensor;

/**
 * A Tensor with the type of the data as FLOAT.
 */
typedef struct {
    /**
     * Metadata of the tensor.
     */
    Tensor* base;
 
    /**
     * Data of the tensor.
     */
    float *tensor;
} FloatTensor;

/**
 * A Tensor with the type of the data as DOUBLE.
 */
typedef struct {
    /**
     * Metadata of the tensor.
     */
    Tensor* base;
 
    /**
     * Data of the tensor.
     */
    double *tensor;
} DoubleTensor;

void freeIntegerTensor(IntegerTensor* tensor);
void freeFloatTensor(FloatTensor* tensor);
void freeDoubleTensor(DoubleTensor* tensor);

size_t countNumberOfDataIndexes(const int dimensions, const int *shape);

IntegerTensor* IntegerTensor_zeros(const int dimensions, const int *shape);
FloatTensor* FloatTensor_zeros(const int dimensions, const int *shape);
DoubleTensor* DoubleTensor_zeros(const int dimensions, const int *shape);

IntegerTensor* IntegerTensor_ones(const int dimensions, const int *shape);
FloatTensor* FloatTensor_ones(const int dimensions, const int *shape);
DoubleTensor* DoubleTensor_ones(const int dimensions, const int *shape);

int *generateDimensionBasedCummulativeJumpTable(const Tensor* tensor);
void IntegerTensor_print(const IntegerTensor* tensor);
void FloatTensor_print(const FloatTensor* tensor);
void DoubleTensor_print(const DoubleTensor* tensor);

Tensor* getTensorBaseByType(const void* tensor, const TensorType type);

#endif