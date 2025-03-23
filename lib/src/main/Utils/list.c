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

#include <stdlib.h>

#include "Utils/list.h"
#include "Error/exceptions.h"

List* createNewList(const size_t initialCapacity) {
    if (initialCapacity <= 0) {
        (void)throwIllegalArgumentException("Initial capacity must be a positive integer.");
        return NULL;
    }

    List* list = (List*)calloc(1, sizeof(List));

    if (list == NULL) {
        (void)throwMemoryAllocationException("At list init.");
        return NULL;
    }

    list->list = (void**)calloc(initialCapacity, sizeof(void*));

    if (list->list == NULL) {
        (void)free(list);
        (void)throwMemoryAllocationException("At list data init.");
        return NULL;
    }

    list->capacity = initialCapacity;
    list->size = 0;
    return list;
}

void rescaleList(List* list, const size_t newSize) {
    void** temp = (void**)realloc(list->list, newSize * sizeof(void*));

    if (temp == NULL) {
        (void)throwMemoryAllocationException("At rescaling list.");
        return;
    }

    list->list = temp;
    list->capacity = newSize;
}

void List_append(List* list, void* ptr) {
    if (list->size + 1 >= list->capacity) {
        (void)rescaleList(list, list->capacity + 1);
    }

    list->list[list->size++] = ptr;
}

void *List_get(List* list, const size_t index) {
    if (index >= list->size) {
        (void)throwIllegalArgumentException("Index is greater than list itself.");
        return NULL;
    }

    return list->list[index];
}

void List_free(List* list) {
    (void)free(list->list);
    (void)free(list);
}