# Python Bindings
Experimental code is experimental.

Install both the C++ end of pybind11 and the python end.

Build with
```
c++ -O2 -Wall -shared -std=c++11 -undefined dynamic_lookup `python3 -m pybind11 --includes` mr_bpnn_2.cpp bpnn.cpp mapreduce.a -o mrbpnn`python3-config --extension-suffix`
```

And import `mrbpnn` from your Python code.
