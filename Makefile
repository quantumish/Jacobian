
CXXFLAGS = -shared -std=c++11 -undefined dynamic_lookup `python3 -m pybind11 --includes` mr_bpnn_2.cpp bpnn.cpp utils.cpp mapreduce.a -o mrbpnn`python3-config --extension-suffix`
all: build

debug: CXXFLAGS += -Wall -U NDEBUG

fast: CXXFLAGS += -O3
fast: build

faster: CXXFLAGS = -shared -std=c++11 -undefined dynamic_lookup `python3 -m pybind11 --includes` mr_bpnn_2.cpp bpnn.cpp utils.cpp mapreduce.a  ${MKLROOT}/lib/libmkl_intel_ilp64.a ${MKLROOT}/lib/libmkl_intel_thread.a ${MKLROOT}/lib/libmkl_core.a -liomp5 -lpthread -lm -ldl -o mrbpnn`python3-config --extension-suffix` -O3 -mavx -mfma -march=native -mfpmath=sse  -DMKL_ILP64 -I${MKLROOT}/include -D EIGEN_USE_MKL_ALL -D NDEBUG
faster: build

tradeoffs: CXXFLAGS = -shared -std=c++11 -undefined dynamic_lookup `python3 -m pybind11 --includes` mr_bpnn_2.cpp bpnn.cpp utils.cpp mapreduce.a  ${MKLROOT}/lib/libmkl_intel_ilp64.a ${MKLROOT}/lib/libmkl_intel_thread.a ${MKLROOT}/lib/libmkl_core.a -liomp5 -lpthread -lm -ldl -o mrbpnn`python3-config --extension-suffix` -O3 -mavx -mfma -march=native -mfpmath=sse -DMKL_ILP64 -I${MKLROOT}/include -ffinite-math-only -fno-math-errno -ffp-contract=fast -D EIGEN_USE_MKL_ALL -D NDEBUG
tradeoffs: build

reckless: CXXFLAGS = -O3
reckless: build

build:
	g++ $(CXXFLAGS)
