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

#ifndef COMPARE_H
#define COMPARE_H

#include "Tensor/tensor.h"

typedef int (*Integer_SearchFunction)(int, int);
typedef int (*Float_SearchFunction)(float, float);
typedef int (*Double_SearchFunction)(double, double);

int Integer_isMin(int a, int b);
int Integer_isMax(int a, int b);

int Float_isMin(float a, float b);
int Float_isMax(float a, float b);

int Double_isMin(double a, double b);
int Double_isMax(double a, double b);

#endif