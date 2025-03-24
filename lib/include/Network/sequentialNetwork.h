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

#ifndef SEQUENTIAL_NETWORK_H
#define SEQUENTIAL_NETWORK_H

#include "Network/layer.h"
#include "Utils/list.h"

typedef struct {
    List* layers;
} SequentialNetwork;

typedef struct {
    const void* layer;
    LayerType type;
} NetworkEntry;


SequentialNetwork* createSequentialNetwork();

void SequentialNetwork_addLayer(const SequentialNetwork* network,
    const void* layer, const LayerType layerType);

void SequentialNetwork_free(SequentialNetwork* network);


IntegerTensor *Integer_SequentialNetwork_forward(const SequentialNetwork* network,
    const IntegerTensor* tensor);

FloatTensor *Float_SequentialNetwork_forward(const SequentialNetwork* network,
    const FloatTensor* tensor);

DoubleTensor *Double_SequentialNetwork_forward(const SequentialNetwork* network,
    const DoubleTensor* tensor);

#endif