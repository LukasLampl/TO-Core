#include <stdio.h>
#include <stdlib.h>

#include "testSuite.h"

void testSuite_assertEquals(const int expected, const int got) {
    if (expected != got) {
        printf("Assertion error!");
        printf("> Expected: %i\n", expected);
        printf("> Got: %i\n", got);
    }
}