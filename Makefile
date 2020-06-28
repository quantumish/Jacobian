
CXXFLAGS: mr_bpnn_2.cpp bpnn.cpp mapreduce.a
	g++ -w -shared -std=c++11 -undefined dynamic_lookup `python3 -m pybind11 --includes` mr_bpnn_2.cpp bpnn.cpp utils.cpp mapreduce.a -o mrbpnn`python3-config --extension-suffix`

all: executable

test: CXXFLAGS = -w -shared -std=c++11 -undefined dynamic_lookup `python3 -m pybind11 --includes` mr_bpnn_2.cpp bpnn.cpp utils.cpp mapreduce.a -o mrbpnn`python3-config --extension-suffix` -O0
test: executable

fast: CXXFLAGS += -w -shared -std=c++11 -undefined dynamic_lookup `python3 -m pybind11 --includes` mr_bpnn_2.cpp bpnn.cpp utils.cpp mapreduce.a -o mrbpnn`python3-config --extension-suffix` -O3
fast: executable

faster: CXXFLAGS += -O3 -mavx -mfma -march=native -mfpmath=sse -D NDEBUG
tradeoffs: CXXFLAGS += -O3
reckless: CXXFLAGS += -O3

executable:
	$(CXX) $(CXXFLAGS)
