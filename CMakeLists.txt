cmake_minimum_required(VERSION 3.10.0)

project(jacobian)

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
  set(COMPILE_FLAGS "${COMPILE_FLAGS} -mavx -O3 -mavx -mfma -march=native -mfpmath=sse -DMKL_ILP64")
elseif (TRADEOFFS)
  set(COMPILE_FLAGS "${COMPILE_FLAGS} -mavx -O3 -mavx -msse2 -msse3 -march=native -mfpmath=sse -DMKL_ILP64 -ffast-math -ffast-math")
elseif (RECKLESS)
  set(COMPILE_FLAGS "${COMPILE_FLAGS} -mavx -O3 -mavx -msse2 -msse3 -march=native -mfpmath=sse -DMKL_ILP64 -D NDEBUG -ffast-math -D RECKLESS")
endif()

set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${COMPILE_FLAGS}")

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})
message("Found Eigen3: ${EIGEN3_INCLUDE_DIR}")

if (PYTHON)
  find_package(pybind11 CONFIG REQUIRED)
  include_directories(${pybind11_INCLUDE_DIRS})
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
  pybind11_add_module(_jacobian ./src/pybind.cpp ./src/bpnn.cpp ./src/utils.cpp)
endif (PYTHON)

if (CXX)
  add_executable(jacobian_cli example.cpp ./src/bpnn.cpp ./src/utils.cpp)
endif (CXX)
