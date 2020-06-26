
all: mr_bpnn_2.cpp bpnn.cpp mapreduce.a
	g++ -w -O2 -shared -std=c++11 -undefined dynamic_lookup `python3 -m pybind11 --includes` mr_bpnn_2.cpp bpnn.cpp utils.cpp mapreduce.a -o mrbpnn`python3-config --extension-suffix`

clean:
	$(RM) mrbpnn
