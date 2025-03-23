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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "Tensor/tensor.h"
#include "Operations/baseOperations.h"
#include "Error/exceptions.h"

typedef int (*IntegerBinaryOperator)(int, int);
typedef float (*FloatBinaryOperator)(float, float);
typedef double (*DoubleBinaryOperator)(double, double);

int Integer_multiply(int a, int b) {return a * b;}
int Integer_add(int a, int b) {return a + b;}
int Integer_subtract(int a, int b) {return a - b;}
int Integer_divide(int a, int b) {return a / b;}

float Float_multiply(float a, float b) {return a * b;}
float Float_add(float a, float b) {return a + b;}
float Float_subtract(float a, float b) {return a - b;}
float Float_divide(float a, float b) {return a / b;}

double Double_multiply(double a, double b) {return a * b;}
double Double_add(double a, double b) {return a + b;}
double Double_subtract(double a, double b) {return a - b;}
double Double_divide(double a, double b) {return a / b;}

/**
 * Returns the element index in the given tensors data for
 * a given indeces array.
 * 
 * <p><b>Example:</b><br>
 * If you want to find the index of [3, 4, 5], you'll need to
 * put in the according tensor and an integer array with the
 * indeces that you're searching for.
 * </p>
 * 
 * @param tensor    The tensor from which to get the index from.
 * @param *indices  Desired indices in the tensor.
 * 
 * @return The element index in the tensors data array.
 */
int getElementIndex(const Tensor* tensor, const int* indices) {
    int index = 0;
    int stride = 1;

    for (int i = tensor->dimensions - 1; i >= 0; i--) {
        index += indices[i] * stride;
        stride *= tensor->shape[i];
    }

    return index;
}

/**
 * Checks whether the dimensions of the given tensors match
 * and whether the shapes match as well.
 * 
 * @param *a            First tensor to check for compatability.
 * @param *b            Second tensor to check against the first.
 * @param *operation    String with the operation type.
 * 
 * @throws IllegalArgumentException - When either the dimension does not match
 * or the shapes are different.
 */
void checkTensorCompatability(const Tensor* a, const Tensor* b, const char *operation) {
    if (a->dimensions != b->dimensions) {
        (void)throwIllegalArgumentException("Can't operate on different shaped tensors!");
    }

    for (int i = 0; i < a->dimensions; i++) {
        if (a->shape[i] != b->shape[i]) {
            const int size = 46 + (size_t)strlen(operation);
            char *buffer = (char*)calloc(size, sizeof(char));

            if (buffer != NULL) {
                (void)snprintf(buffer, size, "To perform %s the tensor shapes must be equal.", operation);
                (void)throwIllegalArgumentException(buffer);
                (void)free(buffer);
            } else {
                (void)throwIllegalArgumentException("The tensor shapes must be equal.");
            }
            return;
        }
    }
}

/**
 * Executes the given operations between tensor a and b and writes the results
 * to the destination tensor.
 * 
 * <p><b>Important:</b><br>
 * This function only allows operations between two identically
 * shaped tensors.
 * </p>
 * 
 * @param *a            Tensor to apply operation.
 * @param *b            Tensor to apply operation.
 * @param *destination  Tensor to write to.
 * @param operation     The operation to execute.
 */
void IntegerTensor_operate(const IntegerTensor* a, const IntegerTensor* b,
    const IntegerTensor* destination, const IntegerBinaryOperator operation) {
    (void)checkTensorCompatability(a->base, b->base, "binary operation");
    (void)checkTensorCompatability(a->base, destination->base, "binary operation");

    const int* data_a = a->data;
    const int* data_b = b->data;
    int* dest = destination->data;
    const int* endPtr = data_a + a->base->dataPoints;

    while (data_a < endPtr) {
        *dest++ = operation((*data_a++), (*data_b++));
    }
}

/**
 * Executes the given operations between tensor a and b and writes the results
 * to the destination tensor.
 * 
 * <p><b>Important:</b><br>
 * This function only allows operations between two identically
 * shaped tensors.
 * </p>
 * 
 * @param *a            Tensor to apply operation.
 * @param *b            Tensor to apply operation.
 * @param *destination  Tensor to write to.
 * @param operation     The operation to execute.
 */
void FloatTensor_operate(const FloatTensor* a, const FloatTensor* b,
    const FloatTensor* destination, const FloatBinaryOperator operation) {
    (void)checkTensorCompatability(a->base, b->base, "binary operation");
    (void)checkTensorCompatability(a->base, destination->base, "binary operation");

    const float* data_a = a->data;
    const float* data_b = b->data;
    float* dest = destination->data;
    const float* endPtr = data_a + a->base->dataPoints;

    while (data_a < endPtr) {
        *dest++ = operation((*data_a++), (*data_b++));
    }
}

/**
 * Executes the given operations between tensor a and b and writes the results
 * to the destination tensor.
 * 
 * <p><b>Important:</b><br>
 * This function only allows operations between two identically
 * shaped tensors.
 * </p>
 * 
 * @param *a            Tensor to apply operation.
 * @param *b            Tensor to apply operation.
 * @param *destination  Tensor to write to.
 * @param operation     The operation to execute.
 */
void DoubleTensor_operate(const DoubleTensor* a, const DoubleTensor* b,
    const DoubleTensor* destination, const DoubleBinaryOperator operation) {
    (void)checkTensorCompatability(a->base, b->base, "binary operation");
    (void)checkTensorCompatability(a->base, destination->base, "binary operation");

    const double* data_a = a->data;
    const double* data_b = b->data;
    double* dest = destination->data;
    const double* endPtr = data_a + a->base->dataPoints;

    while (data_a < endPtr) {
        *dest++ = operation((*data_a++), (*data_b++));
    }
}

/**
 * Multiplies two given tensors and writes the results to a given
 * destination tensor.
 * 
 * <p><b>Important:</b><br>
 * This function only allows multiplication of two identically
 * shaped tensors.
 * </p>
 * 
 * @param *a            Pointer to the multiplier tensor.
 * @param *b            Pointer to the multiplicand tensor.
 * @param *destination  Pointer to the destination tensor in which to write.
 */
void IntegerTensor_multiply(const IntegerTensor* a, const IntegerTensor* b,
    const IntegerTensor* destination) {
    (void)IntegerTensor_operate(a, b, destination, Integer_multiply);
}

/**
 * Divides two given tensors and writes the results to a given
 * destination tensor.
 * 
 * <p><b>Important:</b><br>
 * This function only allows division of two identically
 * shaped tensors.
 * </p>
 * 
 * @param *a            Pointer to the dividend tensor.
 * @param *b            Pointer to the divisor tensor.
 * @param *destination  Pointer to the destination tensor in which to write.
 */
void IntegerTensor_divide(const IntegerTensor* a, const IntegerTensor* b,
    const IntegerTensor* destination) {
    (void)IntegerTensor_operate(a, b, destination, Integer_divide);
}

/**
 * Adds two given tensors and writes the results to a given
 * destination tensor.
 * 
 * <p><b>Important:</b><br>
 * This function only allows addition of two identically
 * shaped tensors.
 * </p>
 * 
 * @param *a            Pointer to the first summand tensor.
 * @param *b            Pointer to the second summand tensor.
 * @param *destination  Pointer to the destination tensor in which to write.
 */
void IntegerTensor_add(const IntegerTensor* a, const IntegerTensor* b,
    const IntegerTensor* destination) {
    (void)IntegerTensor_operate(a, b, destination, Integer_add);
}

/**
 * Subtracts two given tensors and writes the results to a given
 * destination tensor.
 * 
 * <p><b>Important:</b><br>
 * This function only allows subtraction of two identically
 * shaped tensors.
 * </p>
 * 
 * @param *a            Pointer to the minuend tensor.
 * @param *b            Pointer to the subtrahend tensor.
 * @param *destination  Pointer to the destination tensor in which to write.
 */
void IntegerTensor_subtract(const IntegerTensor* a, const IntegerTensor* b,
    const IntegerTensor* destination) {
    (void)IntegerTensor_operate(a, b, destination, Integer_subtract);
}

/**
 * Multiplies a given tensor by a given scalar and writes
 * the results into a given destination tensor.
 * 
 * @param *a            Pointer to the tensor to multiply.
 * @param *scalar       Scalar to multiply the tensor with.
 * @param *destination  Pointer to the destination tensor in which to write.
 */
void IntegerTensor_scalarMultiply(const IntegerTensor* a, const int scalar,
    const IntegerTensor* destination) {
    (void)checkTensorCompatability(a->base, destination->base, "scalar multiply");

    const int* data_a = a->data;
    int* dest = destination->data;
    const int* endPtr = data_a + a->base->dataPoints;

    while (data_a < endPtr) {
        *dest++ = scalar * (*data_a++);
    }
}

/**
 * Multiplies two given tensors and writes the results to a given
 * destination tensor.
 * 
 * <p><b>Important:</b><br>
 * This function only allows multiplication of two identically
 * shaped tensors.
 * </p>
 * 
 * @param *a            Pointer to the multiplier tensor.
 * @param *b            Pointer to the multiplicand tensor.
 * @param *destination  Pointer to the destination tensor in which to write.
 */
void FloatTensor_multiply(const FloatTensor* a, const FloatTensor* b,
    const FloatTensor* destination) {
    (void)FloatTensor_operate(a, b, destination, Float_multiply);
}

/**
 * Divides two given tensors and writes the results to a given
 * destination tensor.
 * 
 * <p><b>Important:</b><br>
 * This function only allows division of two identically
 * shaped tensors.
 * </p>
 * 
 * @param *a            Pointer to the dividend tensor.
 * @param *b            Pointer to the divisor tensor.
 * @param *destination  Pointer to the destination tensor in which to write.
 */
void FloatTensor_divide(const FloatTensor* a, const FloatTensor* b,
    const FloatTensor* destination) {
    (void)FloatTensor_operate(a, b, destination, Float_divide);
}

/**
 * Adds two given tensors and writes the results to a given
 * destination tensor.
 * 
 * <p><b>Important:</b><br>
 * This function only allows addition of two identically
 * shaped tensors.
 * </p>
 * 
 * @param *a            Pointer to the first summand tensor.
 * @param *b            Pointer to the second summand tensor.
 * @param *destination  Pointer to the destination tensor in which to write.
 */
void FloatTensor_add(const FloatTensor* a, const FloatTensor* b,
    const FloatTensor* destination) {
    (void)FloatTensor_operate(a, b, destination, Float_add);
}

/**
 * Subtracts two given tensors and writes the results to a given
 * destination tensor.
 * 
 * <p><b>Important:</b><br>
 * This function only allows subtraction of two identically
 * shaped tensors.
 * </p>
 * 
 * @param *a            Pointer to the minuend tensor.
 * @param *b            Pointer to the subtrahend tensor.
 * @param *destination  Pointer to the destination tensor in which to write.
 */
void FloatTensor_subtract(const FloatTensor* a, const FloatTensor* b,
    const FloatTensor* destination) {
    (void)FloatTensor_operate(a, b, destination, Float_subtract);
}

/**
 * Multiplies a given tensor by a given scalar and writes
 * the results into a given destination tensor.
 * 
 * @param *a            Pointer to the tensor to multiply.
 * @param *scalar       Scalar to multiply the tensor with.
 * @param *destination  Pointer to the destination tensor in which to write.
 */
void FloatTensor_scalarMultiply(const FloatTensor* a, const float scalar,
    const FloatTensor* destination) {
    (void)checkTensorCompatability(a->base, destination->base, "scalar multiply");

    const float* data_a = a->data;
    float* dest = destination->data;
    const float* endPtr = data_a + a->base->dataPoints;

    while (data_a < endPtr) {
        *dest++ = scalar * (*data_a++);
    }
}

/**
 * Multiplies two given tensors and writes the results to a given
 * destination tensor.
 * 
 * <p><b>Important:</b><br>
 * This function only allows multiplication of two identically
 * shaped tensors.
 * </p>
 * 
 * @param *a            Pointer to the multiplier tensor.
 * @param *b            Pointer to the multiplicand tensor.
 * @param *destination  Pointer to the destination tensor in which to write.
 */
void DoubleTensor_multiply(const DoubleTensor* a, const DoubleTensor* b,
    const DoubleTensor* destination) {
    (void)DoubleTensor_operate(a, b, destination, Double_multiply);
}

/**
 * Divides two given tensors and writes the results to a given
 * destination tensor.
 * 
 * <p><b>Important:</b><br>
 * This function only allows division of two identically
 * shaped tensors.
 * </p>
 * 
 * @param *a            Pointer to the dividend tensor.
 * @param *b            Pointer to the divisor tensor.
 * @param *destination  Pointer to the destination tensor in which to write.
 */
void DoubleTensor_divide(const DoubleTensor* a, const DoubleTensor* b,
    const DoubleTensor* destination) {
    (void)DoubleTensor_operate(a, b, destination, Double_divide);
}

/**
 * Adds two given tensors and writes the results to a given
 * destination tensor.
 * 
 * <p><b>Important:</b><br>
 * This function only allows addition of two identically
 * shaped tensors.
 * </p>
 * 
 * @param *a            Pointer to the first summand tensor.
 * @param *b            Pointer to the second summand tensor.
 * @param *destination  Pointer to the destination tensor in which to write.
 */
void DoubleTensor_add(const DoubleTensor* a, const DoubleTensor* b,
    const DoubleTensor* destination) {
    (void)DoubleTensor_operate(a, b, destination, Double_add);
}

/**
 * Subtracts two given tensors and writes the results to a given
 * destination tensor.
 * 
 * <p><b>Important:</b><br>
 * This function only allows subtraction of two identically
 * shaped tensors.
 * </p>
 * 
 * @param *a            Pointer to the minuend tensor.
 * @param *b            Pointer to the subtrahend tensor.
 * @param *destination  Pointer to the destination tensor in which to write.
 */
void DoubleTensor_subtract(const DoubleTensor* a, const DoubleTensor* b,
    const DoubleTensor* destination) {
    (void)DoubleTensor_operate(a, b, destination, Double_subtract);
}

/**
 * Multiplies a given tensor by a given scalar and writes
 * the results into a given destination tensor.
 * 
 * @param *a            Pointer to the tensor to multiply.
 * @param *scalar       Scalar to multiply the tensor with.
 * @param *destination  Pointer to the destination tensor in which to write.
 */
void DoubleTensor_scalarMultiply(const DoubleTensor* a, const double scalar,
    const DoubleTensor* destination) {
    (void)checkTensorCompatability(a->base, destination->base, "scalar multiply");

    const double* data_a = a->data;
    double* dest = destination->data;
    const double* endPtr = data_a + a->base->dataPoints;

    while (data_a < endPtr) {
        *dest++ = scalar * (*data_a++);
    }
}