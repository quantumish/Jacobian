# ML in Parallel

## About
"ML in Parallel" (better name pending) is a work-in-progress machine learning library written in C++ designed to run quickly and in parallel. Parallelization is achieved by utilizing Google's powerful MapReduce framework with [a custom implementation in C](https://github.com/richardfeynmanrocks/mapreduce), and the compiled nature of C++/C as well as the optimizations enabled by gcc's O2 layer enable further speedups. On the same benchmark task (of running a small neural network for 50 epochs on a specified dataset) In some preliminary benchmarks "ML in Parallel" has ran up to 50x faster than a simple Keras program. This library is also easily accessible

## Benchmark Info
Coming soon: A detailed rundown of the speed of "ML in Parallel" vs popular machine learning libraries for Python (and eventually comparisons to C++ libraries as well) as well as a handy and flexible Python script for creating benchmark graphs on the fly.

## Testing/Demo
As of now, a demo of a neural network with feedforward and backpropagation is available (as well as a Keras demo for reference). **Note: You'll need to add the `mapreduce` archive file to the project directory, which can be found [here](https://github.com/richardfeynmanrocks/mapreduce).**

### Sequential
See the sequential demo by including the header file in you project, calling `demo`, and building/running with the following commands:
```
g++ YOUR_FILE bpnn.cpp mapreduce.a -O2 -o bpnn -std=c++11 -w
./bpnn
```

### Parallel
Similarly, build and run the parallel demo with these commands:
```
g++ mr_bpnn_2.cpp bpnn.cpp mapreduce.a -O2 -o bpnn -std=c++11 -w
./bpnn NUM_PARALLEL_NETWORKS PATH_TO_DATA YOUR_PUBLIC_IP 1
```

### Other
Compare these demonstrations with a sample Keras demo by running `python kerasdemo.py`

## Usage
As of now "ML in Parallel" is not fully fit for usage inside code.

### Python Bindings
This feature is largely experimental but is the preferred way to demo as of now.

1. Install both the C++ end of pybind11 and the python end.
2. Build with `make` or the much uglier alternative:
```
c++ -w -O2 -Wall -shared -std=c++11 -undefined dynamic_lookup `python3 -m pybind11 --includes` mr_bpnn_2.cpp bpnn.cpp mapreduce.a -o mrbpnn`python3-config --extension-suffix`
```
3. Copy the `mrbpnn.cpython-37m-darwin.so` file into your personal project directory.
4. Import `mrbpnn` from your Python code and use it.
5. Profit.

