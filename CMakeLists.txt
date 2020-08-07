cmake_minimum_required(VERSION 3.10.0)

set(CMAKE_CXX_COMPILER "g++")

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(COMPILE_FLAGS, "-w")

if (!FAST)
  set(COMPILE_FLAGS "${COMPILE_FLAGS} -mavx -O3")
endif (!FAST)

if (!FASTER)
  set(COMPILE_FLAGS "${COMPILE_FLAGS} -mavx -O3 -ffast-math")
endif (!FASTER)

if (!TRADEOFFS)
  set(COMPILE_FLAGS "${COMPILE_FLAGS} -mavx -O3 -ffast-math -D RECKLESS")
endif (!TRADEOFFS)

if (!AVX)
  set(COMPILE_FLAGS "${COMPILE_FLAGS} -mavx -D AVX")
endif (!AVX)

set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${COMPILE_FLAGS}")

project (jacobian)
find_package(pybind11 CONFIG REQUIRED)
include_directories(${pybind11_INCLUDE_DIRS})
#add_library(utils ./src/utils.cpp)
#add_library(bpnn ./src/bpnn.cpp)

pybind11_add_module(jacobian example.cpp ./src/bpnn.cpp ./src/utils.cpp)
#target_link_libraries(jacobian PUBLIC utils bpnn)

enable_testing()
add_test(NAME "Valid run" COMMAND jacobian 16 50)
add_test(NAME "Large batch" COMMAND jacobian 1000 50)
add_test(NAME "Small batch" COMMAND jacobian 1 50)

