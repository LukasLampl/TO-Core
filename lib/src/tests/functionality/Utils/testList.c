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
#include "Utils/list.h"

#include "Tests/testUtil.h"

void testList_001() {
    List* list = createNewList(16);
    int data_1 = 56;
    char data_2[] = "String";
    double data_3 = 3.141;

    List_append(list, &data_1);
    List_append(list, &data_2);
    List_append(list, &data_3);

    int rec_1 = *(int*)List_get(list, 0);
    char *rec_2 = (char*)List_get(list, 1);
    double rec_3 = *(double*)List_get(list, 2);

    printf("Rec1: %i\n", rec_1);
    printf("Rec2: %s\n", rec_2);
    printf("Rec3: %f\n", rec_3);
}