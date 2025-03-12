# Compiler and flags
CC = gcc
CFLAGS = -Wall -Werror -Wpedantic -Ilib/include

# Source files (all .c files in subdirectories)
SRC_FILES = $(wildcard lib/src/main/*.c \
			lib/src/main/Error/*.c \
			lib/src/main/Operations/*.c \
			lib/src/main/Tensor/*.c \
			lib/src/tests/*.c \
			lib/src/tests/profile/*.c \
			lib/src/tests/functionality/*.c \
			lib/src/tests/profile/Integer/*.c \
			lib/src/tests/functionality/Integer/*.c)

# Object files (replace .c with .o for each source file)
OBJ_FILES = $(SRC_FILES:.c=.o)

# Output executable
TARGET = TO-Core

# Default target to build
all: $(TARGET)

# Rule to link object files into the final executable
$(TARGET): $(OBJ_FILES)
	$(CC) $(OBJ_FILES) -o $(TARGET)

# Rule to compile each source file into an object file
$(OBJ_FILES): %.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Clean up object and executable files
clean:
	rm -f $(OBJ_FILES) $(TARGET)
