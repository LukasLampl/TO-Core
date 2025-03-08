cmake_minimum_required(VERSION 3.10)
project(TO-Core)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED True)

include_directories(lib/include)

add_executable(output
lib/src/main/globals.c
lib/src/main/Error/exceptions.c
lib/src/main/Tensor/tensor.c
lib/src/main/Operations/tensorOperations.c
lib/src/main/Operations/Integer/integerTensorOperations.c
)