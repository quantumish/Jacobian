cmake_minimum_required(VERSION 3.10.0)

set(CMAKE_CXX_COMPILER "g++")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (DEBUG)
  set(COMPILE_FLAGS, "-Wall")
else()
  set(COMPILE_FLAGS, "-w")
endif()
  
if (FAST)
  set(COMPILE_FLAGS "${COMPILE_FLAGS} -O3")
elseif (FASTER)
  set(COMPILE_FLAGS "${COMPILE_FLAGS} -mavx -O3 -mavx -mfma -march=native -mfpmath=sse -fno-pic -DMKL_ILP64")
elseif (TRADEOFFS)
  set(COMPILE_FLAGS "${COMPILE_FLAGS} -mavx -O3 -mavx -msse2 -msse3 -march=native -mfpmath=sse -DMKL_ILP64 -fno-pic -ffast-math -ffast-math")
elseif (RECKLESS)
  set(COMPILE_FLAGS "${COMPILE_FLAGS} -mavx -O3 -mavx -msse2 -msse3 -march=native -mfpmath=sse -DMKL_ILP64 -fno-pic -ffast-math -D NDEBUG -ffast-math -D RECKLESS")
endif()


set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${COMPILE_FLAGS} -w -llz4")

# if (PYTHON)
#   project(jacobian)
#   find_package(pybind11 CONFIG REQUIRED)
#   include_directories(${pybind11_INCLUDE_DIRS})
#   pybind11_add_module(_jacobian example.cpp ./src/bpnn.cpp ./src/utils.cpp)
# endif (PYTHON)

if (CXX)
  project(jacobian_cli)
  add_executable(jacobian_cli example.cpp ./src/bpnn.cpp ./src/utils.cpp)
endif (CXX)
