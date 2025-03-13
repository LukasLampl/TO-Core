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

#include "globals.h"
#include "Error/exceptions.h"

/**
 * Holds the display names of each exception type.
 * The according display name can be acquired by using the
 * ExceptionType enums index.
 */
static const char *EXCEPTION_PREFIX[3] = {
    "MemoryAllocationException",
    "NullPointerException",
    "IllegalArgumentException"
};

/**
 * Throws an base exception with the given type and message.
 * It'll print the display name of the exception type, followed
 * by the message itself. Finally the program will exit, except
 * when unit testing is on.
 * 
 * @param type      Exception type that is thrown.
 * @param *message  The message to display as the exception.
 */
void throwException(ExceptionType type, char *message) {
    (void)printf("%s: %s\n", EXCEPTION_PREFIX[type], message);
    
    if (!ENV_UNIT_TESTING) {
        (void)exit(-1);
    }
}

/**
 * Throws an `MemoryAllocationException` with the given message.
 * This should be used when allocating memory and checking for
 * NULL.
 * 
 * @param *message  The message to display.
 * 
 * @see #throwException(ExceptionType type, char *message)
 */
void throwMemoryAllocationException(char *message) {
    (void)throwException(MemoryAllocationException, message);
}

/**
 * Throws an `NullPointerException` with the given message.
 * This should be used to check for NULL values that could
 * impact the programs output.
 * 
 * @param *message  The message to display.
 * 
 * @see #throwException(ExceptionType type, char *message)
 */
void throwNullPointerException(char *message) {
    (void)throwException(NullPointerException, message);
}

/**
 * Throws an `IllegalArgumentException` with the given message.
 * This should be used when checking arguments of a function
 * to validate the given values.
 * 
 * @param *message  The message to display.
 * 
 * @see #throwException(ExceptionType type, char *message)
 */
void throwIllegalArgumentException(char *message) {
    (void)throwException(IllegalArgumentException, message);
}