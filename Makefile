#
#  Makefile
#  Jacobian
#
#  Created by David Freifeld
#
# ---------
# TODO:
# Start using CMake to make this less of a headache?
# ---------

GEN_FLAGS = -fpic

CXX = icpc
CXXFLAGS = -shared -std=c++17 -undefined dynamic_lookup `python3 -m pybind11 --includes` ./src/pybind.cpp ./src/bpnn.cpp ./src/utils.cpp mapreduce.a -o mrbpnn`python3-config --extension-suffix`
#LDFLAGS = -L/usr/local/opt/llvm/lib -lc++

all: compile

debug: $(GEN_FLAGS) = -Wall -U NDEBUG

test: CXXFLAGS = -std=c++17 example.cpp ./src/bpnn.cpp ./src/utils.cpp mapreduce.a -o bpnnexec -O3 -mavx -mfma -march=native -mfpmath=sse -fno-pic -DMKL_ILP64 -D NDEBUG
test: LDFLAGS = -lpthread -lm -ldl
test: compile

fast: CXXFLAGS += $(GEN_FLAGS) -O3
fast: compile

faster: CXXFLAGS = -shared -std=c++17 -undefined dynamic_lookup `python3 -m pybind11 --includes` ./src/pybind.cpp ./src/bpnn.cpp ./src/utils.cpp mapreduce.a -o mrbpnn`python3-config --extension-suffix` -O3 -mavx -msse2 -msse3 -fopenmp -march=native -mfpmath=sse -fno-pic -DMKL_ILP64 -D NDEBUG -D EIGEN_USE_BLAS ${MKLROOT}/lib/libmkl_intel_ilp64.a -m64 -I${MKLROOT}/include ${MKLROOT}/lib/libmkl_intel_thread.a ${MKLROOT}/lib/libmkl_core.a -liomp5 -lpthread -lm -ldl
faster: LDFLAGS = -lpthread -lm -ldl -lblas
faster: compile

tradeoffs: CXXFLAGS = -shared -std=c++17 -undefined dynamic_lookup `python3 -m pybind11 --includes` ./src/pybind.cpp ./src/bpnn.cpp ./src/utils.cpp mapreduce.a -o mrbpnn`python3-config --extension-suffix` -O3 -mavx -msse2 -msse3 -march=native -mfpmath=sse -DMKL_ILP64 -fno-pic -ffast-math -D NDEBUG #-qopt-report=5 -qopt-report-file=report
tradeoffs: compile


reckless: CXXFLAGS = -O3
reckless: compile

compile:
	$(CXX) $(CXXFLAGS) $(LDFLAGS) && rm ./mrbpnn/mrbpnn.cpython-37m-darwin.so ; cp ./mrbpnn.cpython-37m-darwin.so ./mrbpnn/mrbpnn.cpython-37m-darwin.s ; rm ./scripts/mrbpnn.cpython-37m-darwin.so ; cp ./mrbpnn.cpython-37m-darwin.so ./scripts/mrbpnn.cpython-37m-darwin.so
