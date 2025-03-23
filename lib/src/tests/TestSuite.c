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

#include "testSuite.h"

void printError(const double expected, const double got) {
    printf("Assertion error!\n");
    printf("> Expected: %f\n", expected);
    printf("> Got: %f\n", got);
}

void testSuite_assertEquals(const int expected, const int got) {
    if (expected != got) {
        printError(expected, got);
    }
}

void testSuite_assertInBetween(const double value, const double min, const double max) {
    if (value > max) {
        printError(max, value);
    } else if (value < min) {
        printError(min, value);
    }
}