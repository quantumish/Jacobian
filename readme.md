
# ML in Parallel

## About
"ML in Parallel" (better name pending) is a work-in-progress machine learning library written in C++ designed to run quickly and in parallel. Parallelization is achieved by utilizing Google's powerful MapReduce framework with [a custom implementation in C](https://github.com/richardfeynmanrocks/mapreduce), and the compiled nature of C++/C as well as the optimizations enabled by gcc's O2 layer enable further speedups. On the same benchmark task (of running a small neural network for 50 epochs on a specified dataset) "ML in Parallel" can run up to 45x faster than a simple Keras program. Python bindings are being developed for this network to allow similar ease of usage to Keras while maintaining the enhanced speed.

## Benchmark Info
Coming soon: A detailed rundown of the speed of "ML in Parallel" vs popular machine learning libraries for Python (and eventually comparisons to C++ libraries as well) as well as a handy and flexible Python script for creating benchmark graphs on the fly.

## Testing/Demo
As of now, a demo of a neural network with feedforward and backpropagation is available (as well as a Keras demo for reference). **Note: You'll need to add the `mapreduce` archive file to the project directory, which can be found [here](https://github.com/richardfeynmanrocks/mapreduce).**

### Sequential
Build and run the sequential demo with the following commands:
```
g++ mr_bpnn_2.cpp bpnn.cpp mapreduce.a -O2 -o bpnn -std=c++11 -w
./bpnn NUM_PARALLEL_NETWORKS PATH_TO_DATA YOUR_PUBLIC_IP 1
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
As of now "ML in Parallel" is not fit for usage inside code.
However:
- Initial prototype code is being cleaned up so as to make it more usable.
- Python bindings with `pybind11` are in development for improved usage.
