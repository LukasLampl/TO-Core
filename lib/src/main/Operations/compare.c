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

#include "Tensor/tensor.h"
#include "Operations/compare.h"

#define true 1
#define false 0

/**
 * Checks whether the given `a` is smaller than `b`.
 * 
 * @param a     First number to compare.
 * @param b     Second number to compare.
 * 
 * @return
 * <ul>
 * <li>`true` - When `a` < `b`.
 * <li>`false` - When `a` > `b`.
 * </ul>
 */
int Integer_isMin(int a, int b) {
    return a < b ? true : false;
}

/**
 * Checks whether the given `a` is greater than `b`.
 * 
 * @param a     First number to compare.
 * @param b     Second number to compare.
 * 
 * @return
 * <ul>
 * <li>`true` - When `a` > `b`.
 * <li>`false` - When `a` < `b`.
 * </ul>
 */
int Integer_isMax(int a, int b) {
    return a > b ? true : false;
}

/**
 * Checks whether the given `a` is smaller than `b`.
 * 
 * @param a     First number to compare.
 * @param b     Second number to compare.
 * 
 * @return
 * <ul>
 * <li>`true` - When `a` < `b`.
 * <li>`false` - When `a` > `b`.
 * </ul>
 */
int Float_isMin(float a, float b) {
    return a < b ? true : false;
}

/**
 * Checks whether the given `a` is greater than `b`.
 * 
 * @param a     First number to compare.
 * @param b     Second number to compare.
 * 
 * @return
 * <ul>
 * <li>`true` - When `a` > `b`.
 * <li>`false` - When `a` < `b`.
 * </ul>
 */
int Float_isMax(float a, float b) {
    return a > b ? true : false;
}

/**
 * Checks whether the given `a` is smaller than `b`.
 * 
 * @param a     First number to compare.
 * @param b     Second number to compare.
 * 
 * @return
 * <ul>
 * <li>`true` - When `a` < `b`.
 * <li>`false` - When `a` > `b`.
 * </ul>
 */
int Double_isMin(double a, double b) {
    return a < b ? true : false;
}

/**
 * Checks whether the given `a` is greater than `b`.
 * 
 * @param a     First number to compare.
 * @param b     Second number to compare.
 * 
 * @return
 * <ul>
 * <li>`true` - When `a` > `b`.
 * <li>`false` - When `a` < `b`.
 * </ul>
 */
int Double_isMax(double a, double b) {
    return a > b ? true : false;
}