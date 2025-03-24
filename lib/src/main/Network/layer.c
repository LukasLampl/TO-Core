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
#include "Network/layer.h"

Layer *createLayer(const TensorType tensorType) {
    Layer* layer = (Layer*)calloc(1, sizeof(Layer));

    if (layer == NULL) {
        (void)throwMemoryAllocationException("While trying to create new layer.");
        return NULL;
    }

    layer->inputType = tensorType;
    return layer;
}

void freeLayer(Layer* layer) {
    if (layer != NULL) {
        (void)free(layer);
    }
}