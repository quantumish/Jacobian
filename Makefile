
CXXFLAGS: mr_bpnn_2.cpp bpnn.cpp mapreduce.a
	g++ -w -shared -std=c++11 -undefined dynamic_lookup `python3 -m pybind11 --includes` mr_bpnn_2.cpp bpnn.cpp utils.cpp mapreduce.a -o mrbpnn`python3-config --extension-suffix`

all: executable

test: CXXFLAGS = -w -shared -std=c++11 -undefined dynamic_lookup `python3 -m pybind11 --includes` mr_bpnn_2.cpp bpnn.cpp utils.cpp mapreduce.a -o mrbpnn`python3-config --extension-suffix` -O0
test: build

fast: CXXFLAGS = -w -shared -std=c++11 -undefined dynamic_lookup `python3 -m pybind11 --includes` mr_bpnn_2.cpp bpnn.cpp utils.cpp mapreduce.a -o mrbpnn`python3-config --extension-suffix` -O3
fast: build

faster: CXXFLAGS = -w -shared -std=c++11 -undefined dynamic_lookup `python3 -m pybind11 --includes` mr_bpnn_2.cpp bpnn.cpp utils.cpp mapreduce.a -o mrbpnn`python3-config --extension-suffix` -O3 -mavx -mfma -march=native -mfpmath=sse -D NDEBUG
faster: build

tradeoffs: CXXFLAGS = -w -shared -std=c++11 -undefined dynamic_lookup `python3 -m pybind11 --includes` mr_bpnn_2.cpp bpnn.cpp utils.cpp mapreduce.a -o mrbpnn`python3-config --extension-suffix` -O3 -mavx -mfma -march=native -mfpmath=sse -ffinite-math-only -fno-math-errno -ffp-contract=fast -D NDEBUG
tradeoffs: build

reckless: CXXFLAGS = -O3
reckless: build

build:
	$(CXX) $(CXXFLAGS)
