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

#include "Error/exceptions.h"
#include "Utils/list.h"
#include "Network/sequentialNetwork.h"

#include "Operations/convolution.h"
#include "Operations/activation.h"

#define true 1
#define false 0

SequentialNetwork* createSequentialNetwork() {
    SequentialNetwork* network = (SequentialNetwork*)calloc(1, sizeof(SequentialNetwork));
    List* list = (List*)createNewList(16);

    if (network == NULL || list == NULL) {
        if (list != NULL) (void)free(list);
        if (network != NULL) (void)free(network);
        (void)throwMemoryAllocationException("While trying to create a SequentialNetwork.");
        return NULL;
    }

    network->layers = list;
    return network;
}

void SequentialNetwork_addLayer(const SequentialNetwork* network,
    const void* layer, const LayerType layerType) {
    NetworkEntry* entry = (NetworkEntry*)calloc(1, sizeof(NetworkEntry));

    if (entry == NULL) {
        (void)throwMemoryAllocationException("At entry allocation for SequentialNetwork.");
        return;
    }

    entry->layer = layer;
    entry->type = layerType;
    (void)List_append(network->layers, entry);
}

void SequentialNetwork_free(SequentialNetwork* network) {
    for (int i = 0; i < network->layers->size; i++) {
        NetworkEntry* entry = (NetworkEntry*)List_get(network->layers, i);
        void* layer = (void*)entry->layer;

        switch (entry->type) {
        case CONVOLUTION:
            (void)ConvolutionLayer_free(layer);
            break;
        case ACTIVATION:
            (void)ActivationLayer_free(layer);
            break;
        }

        (void)free(entry);
    }

    (void)List_free(network->layers);
    (void)free(network);
}

void* executeLayer(const NetworkEntry* entry, void* input) {
    switch (entry->type) {
    case CONVOLUTION: {
        ConvolutionLayer* layer = (ConvolutionLayer*)entry->layer;
        (void)ConvolutionLayer_forward(layer, input);
        return layer->base->destination;
    case ACTIVATION: {
        ActivationLayer* layer = (ActivationLayer*)entry->layer;
        (void)ActivationLayer_forward(layer, input);
        return layer->base->destination;
    }
    }
    }

    return input;
}

void *forward(const SequentialNetwork* network, void* tensor) {
    void* currentTensor = tensor;

    for (int i = 0; i < network->layers->size; i++) {
        NetworkEntry* entry = (NetworkEntry*)List_get(network->layers, i);
        currentTensor = (void*)executeLayer(entry, currentTensor);
    }

    return currentTensor;
}

IntegerTensor *Integer_SequentialNetwork_forward(const SequentialNetwork* network,
    const IntegerTensor* tensor) {
    return (IntegerTensor*)forward(network, (void*)tensor);
}

FloatTensor *Float_SequentialNetwork_forward(const SequentialNetwork* network,
    const FloatTensor* tensor) {
    return (FloatTensor*)forward(network, (void*)tensor);
}

DoubleTensor *Double_SequentialNetwork_forward(const SequentialNetwork* network,
    const DoubleTensor* tensor) {
    return (DoubleTensor*)forward(network, (void*)tensor);
}