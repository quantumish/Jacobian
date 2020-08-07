cmake_minimum_required(VERSION 3.10.0)
find_package(pybind11 CONFIG REQUIRED)
include_directories(${pybind11_INCLUDE_DIRS})

project (jacobian)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
#add_library(utils ./src/utils.cpp)
#add_library(bpnn ./src/bpnn.cpp)

SET(GCC_COMPILE_FLAGS "-w")
SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${GCC_COMPILE_FLAGS}")

add_executable(jacobian example.cpp ./src/bpnn.cpp ./src/utils.cpp)
#target_link_libraries(jacobian PUBLIC utils bpnn)

enable_testing()
add_test(NAME "Valid run" COMMAND jacobian 16 50)
add_test(NAME "Large batch" COMMAND jacobian 1000 50)
add_test(NAME "Small batch" COMMAND jacobian 1 50)

