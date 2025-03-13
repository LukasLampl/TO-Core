#ifndef TENSOR_H
#define TENSOR_H

#include <stdlib.h>

typedef struct {
    int dimensions;
    int* shape;
    size_t dataPoints;
} Tensor;

typedef struct {
    Tensor* base;
    int *tensor;
} IntegerTensor;

typedef struct {
    Tensor* base;
    float *tensor;
} FloatTensor;

typedef struct {
    Tensor* base;
    double *tensor;
} DoubleTensor;

IntegerTensor* createIntegerTensor(const int dimensions, const int *shape);
FloatTensor* createFloatTensor(const int dimensions, const int *shape);
DoubleTensor* createDoubleTensor(const int dimensions, const int *shape);

void IntegerTensor_print(const IntegerTensor* tensor);

void freeIntegerTensor(IntegerTensor* tensor);
void freeFloatTensor(FloatTensor* tensor);
void freeDoubleTensor(DoubleTensor* tensor);

#endif