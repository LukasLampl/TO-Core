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

#include "testSuite.h"
#include "Operations/convolution.h"
#include "Operations/activation.h"
#include "Network/sequentialNetwork.h"
#include "Network/layer.h"
#include "Tensor/tensor.h"

#include "Tests/testNetwork.h"

void test_SN_Convolution_001() {
    printf("Test_SN_Convolution_001...\n");
    int shape_kernel[] = {3, 3};
    int shape_tensor[] = {4, 4};
    DoubleTensor* kernel = DoubleTensor_zeros(2, shape_kernel);
    DoubleTensor* tensor = DoubleTensor_zeros(2, shape_tensor);

    tensor->data[0] = 1.0;      tensor->data[1] = 7.0;      tensor->data[2] = 9.0;      tensor->data[3] = 2.3;
    tensor->data[4] = 4.0;      tensor->data[5] = 1.8;      tensor->data[6] = 6.5;      tensor->data[7] = 4.5;
    tensor->data[8] = 3.0;      tensor->data[9] = 3.4;      tensor->data[10] = 7.3;     tensor->data[11] = 8.7;
    tensor->data[12] = 1.2;     tensor->data[13] = 1.6;     tensor->data[14] = 1.4;     tensor->data[15] = 2.3;

    kernel->data[0] = 1.0;      kernel->data[1] = 3.4;      kernel->data[2] = 1.1;
    kernel->data[3] = 8.4;      kernel->data[4] = 7.6;      kernel->data[5] = 3.2;
    kernel->data[6] = 3.7;      kernel->data[7] = 4.0;      kernel->data[8] = 5.1;

    SequentialNetwork* net = createSequentialNetwork();
    
    ConvolutionLayer* layer = Double_createConvolutionLayer(kernel, NULL, 1);
    
    SequentialNetwork_addLayer(net, layer, CONVOLUTION);

    DoubleTensor* result = Double_SequentialNetwork_forward(net, tensor);
    
    testSuite_assertEquals(164.71, result->data[0]);
    testSuite_assertEquals(205.2, result->data[1]);
    testSuite_assertEquals(109.65, result->data[2]);
    testSuite_assertEquals(163.98, result->data[3]);

    SequentialNetwork_free(net);
    freeDoubleTensor(tensor);
    freeDoubleTensor(kernel);
    printf("> Pass\n\n");
}

void test_SN_Convolution_002() {
    printf("Test_SN_Convolution_002...\n");
    int shape_kernel_1[] = {3, 3};
    int shape_kernel_2[] = {2, 2};
    int shape_tensor[] = {4, 4};
    DoubleTensor* kernel_1 = DoubleTensor_zeros(2, shape_kernel_1);
    DoubleTensor* kernel_2 = DoubleTensor_zeros(2, shape_kernel_2);
    DoubleTensor* tensor = DoubleTensor_zeros(2, shape_tensor);

    tensor->data[0] = 1.0;      tensor->data[1] = 7.0;      tensor->data[2] = 9.0;      tensor->data[3] = 2.3;
    tensor->data[4] = 4.0;      tensor->data[5] = 1.8;      tensor->data[6] = 6.5;      tensor->data[7] = 4.5;
    tensor->data[8] = 3.0;      tensor->data[9] = 3.4;      tensor->data[10] = 7.3;     tensor->data[11] = 8.7;
    tensor->data[12] = 1.2;     tensor->data[13] = 1.6;     tensor->data[14] = 1.4;     tensor->data[15] = 2.3;

    kernel_1->data[0] = 1.0;      kernel_1->data[1] = 3.4;      kernel_1->data[2] = 1.1;
    kernel_1->data[3] = 8.4;      kernel_1->data[4] = 7.6;      kernel_1->data[5] = 3.2;
    kernel_1->data[6] = 3.7;      kernel_1->data[7] = 4.0;      kernel_1->data[8] = 5.1;

    kernel_2->data[0] = 6.0;      kernel_2->data[1] = 9.9;
    kernel_2->data[2] = 3.2;      kernel_2->data[3] = 5.3;

    SequentialNetwork* net = createSequentialNetwork();
    
    ConvolutionLayer* layer_1 = Double_createConvolutionLayer(kernel_1, NULL, 1);
    ConvolutionLayer* layer_2 = Double_createConvolutionLayer(kernel_2, NULL, 1);
    
    SequentialNetwork_addLayer(net, layer_1, CONVOLUTION);
    SequentialNetwork_addLayer(net, layer_2, CONVOLUTION);

    DoubleTensor* result = Double_SequentialNetwork_forward(net, tensor);
    
    testSuite_assertEquals(4239.714, result->data[0]);
    
    SequentialNetwork_free(net);
    freeDoubleTensor(tensor);
    freeDoubleTensor(kernel_1);
    freeDoubleTensor(kernel_2);
    printf("> Pass\n\n");
}

void test_SN_Activation_001() {
    printf("Test_SN_Activation_001...\n");
    int shape[] = {3, 3};
    DoubleTensor* tensor = DoubleTensor_zeros(2, shape);

    for (int i = 0; i < tensor->base->dataPoints; i++) {
        tensor->data[i] = i;
    }

    ActivationLayer* layer = Double_createActivationLayer(SIGMOID, 0);
    SequentialNetwork* net = createSequentialNetwork();
    SequentialNetwork_addLayer(net, layer, ACTIVATION);

    DoubleTensor* result = Double_SequentialNetwork_forward(net, tensor);

    testSuite_assertEquals(0.5, result->data[0]);
    testSuite_assertEquals(0.731059, result->data[1]);
    testSuite_assertEquals(0.880797, result->data[2]);
    testSuite_assertEquals(0.952574, result->data[3]);
    testSuite_assertEquals(0.982014, result->data[4]);
    testSuite_assertEquals(0.993307, result->data[5]);
    testSuite_assertEquals(0.997527, result->data[6]);
    testSuite_assertEquals(0.999089, result->data[7]);
    testSuite_assertEquals(0.999665, result->data[8]);

    SequentialNetwork_free(net);
    freeDoubleTensor(tensor);
    printf("> Pass\n\n");
}

void test_SN_Activation_002() {
    printf("Test_SN_Activation_002...\n");
    int shape_kernel[] = {3, 3};
    int shape_tensor[] = {4, 4};
    DoubleTensor* kernel = DoubleTensor_zeros(2, shape_kernel);
    DoubleTensor* tensor = DoubleTensor_zeros(2, shape_tensor);

    tensor->data[0] = 1.0;      tensor->data[1] = 7.0;      tensor->data[2] = 9.0;      tensor->data[3] = 2.3;
    tensor->data[4] = 4.0;      tensor->data[5] = 1.8;      tensor->data[6] = 6.5;      tensor->data[7] = 4.5;
    tensor->data[8] = 3.0;      tensor->data[9] = 3.4;      tensor->data[10] = 7.3;     tensor->data[11] = 8.7;
    tensor->data[12] = 1.2;     tensor->data[13] = 1.6;     tensor->data[14] = 1.4;     tensor->data[15] = 2.3;

    kernel->data[0] = 1.0;      kernel->data[1] = 3.4;      kernel->data[2] = 1.1;
    kernel->data[3] = 8.4;      kernel->data[4] = 7.6;      kernel->data[5] = 3.2;
    kernel->data[6] = 3.7;      kernel->data[7] = 4.0;      kernel->data[8] = 5.1;

    SequentialNetwork* net = createSequentialNetwork();
    
    ConvolutionLayer* layer_1 = Double_createConvolutionLayer(kernel, NULL, 1);
    ActivationLayer* layer_2 = Double_createActivationLayer(TANH, 0);

    SequentialNetwork_addLayer(net, layer_1, CONVOLUTION);
    SequentialNetwork_addLayer(net, layer_2, ACTIVATION);

    DoubleTensor* result = Double_SequentialNetwork_forward(net, tensor);
    
    testSuite_assertEquals(1, result->data[0]);
    testSuite_assertEquals(1, result->data[1]);
    testSuite_assertEquals(1, result->data[2]);
    testSuite_assertEquals(1, result->data[3]);

    SequentialNetwork_free(net);
    freeDoubleTensor(tensor);
    freeDoubleTensor(kernel);
    printf("> Pass\n\n");
}