# Compiler and flags
CC = gcc
CFLAGS = -Wall -Werror -Wpedantic -Ilib/include -O3

# Source files (all .c files in subdirectories)
SRC_FILES = $(wildcard lib/src/main/*.c \
			lib/src/main/Error/*.c \
			lib/src/main/Operations/*.c \
			lib/src/main/Tensor/*.c \
			lib/src/main/Network/*.c \
			lib/src/main/Utils/*.c \
			lib/src/tests/*.c \
			lib/src/tests/profile/*.c \
			lib/src/tests/functionality/*.c \
			lib/src/tests/profile/Integer/*.c \
			lib/src/tests/functionality/Integer/*.c \
			lib/src/tests/functionality/Utils/*.c \
			lib/src/tests/functionality/Network/*.c)

# Object files (replace .c with .o for each source file)
OBJ_FILES = $(SRC_FILES:.c=.o)
DEP_FILES = $(OBJ_FILES:.o=.d)  # Dependency files

# Output executable
TARGET = TO-Core

# Default target to build
all: $(TARGET)

# Rule to link object files into the final executable (Added $(CFLAGS) for OpenMP linking)
$(TARGET): $(OBJ_FILES)
	$(CC) $(CFLAGS) $(OBJ_FILES) -o $(TARGET)

# Rule to compile each source file into an object file (Generates dependency files)
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Include dependency files if they exist (Ensures proper rebuilds on header changes)
-include $(DEP_FILES)

# Clean up object and executable files
clean:
	rm -f $(OBJ_FILES) $(DEP_FILES) $(TARGET)
