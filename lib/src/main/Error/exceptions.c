#include <stdio.h>
#include <stdlib.h>

#include "globals.h"
#include "Error/exceptions.h"

static const char *EXCEPTION_PREFIX[3] = {
    "MemoryAllocationException",
    "NullPointerException",
    "IllegalArgumentException"
};

void throwException(ExceptionType type, char *message) {
    (void)printf("%s: %s\n", EXCEPTION_PREFIX[type], message);
    
    if (!ENV_UNIT_TESTING) {
        (void)exit(-1);
    }
}

void throwMemoryAllocationException(char *message) {
    throwException(MemoryAllocationException, message);
}

void throwNullPointerException(char *message) {
    throwException(NullPointerException, message);
}

void throwIllegalArgumentException(char *message) {
    throwException(IllegalArgumentException, message);
}