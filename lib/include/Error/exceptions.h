#ifndef EXCEPTIONS_H
#define EXCEPTIONS_H

typedef enum {
    MemoryAllocationException = 0,
    NullPointerException = 1,
    IllegalArgumentException = 2
} ExceptionType;

void throwMemoryAllocationException(char *message);
void throwNullPointerException(char *message);
void throwIllegalArgumentException(char *message);

#endif