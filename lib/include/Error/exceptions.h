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

#ifndef EXCEPTIONS_H
#define EXCEPTIONS_H

/**
 * Types of exceptions and their according index.
 * The index is used for a display name dictionary.
 */
typedef enum {
    MemoryAllocationException = 0,
    NullPointerException = 1,
    IllegalArgumentException = 2
} ExceptionType;

void throwMemoryAllocationException(char *message);
void throwNullPointerException(char *message);
void throwIllegalArgumentException(char *message);

#endif