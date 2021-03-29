  <!-- readme.md -->
  <!-- Jacobian -->

  <!-- Created by David Freifeld -->
  <!-- Markdown has the worst comment syntax I've ever seen. Seriously, what is this. TODO: Migrate this README to Org Mode. -->

![Banner](./pictures/banner.png)

## About
Jacobian is a work-in-progress machine learning library written in C++ designed to run as fast as possible while still being simple to use. Jacobian is accessible via Python and enables you to write models that train faster with the same amount of code. As of now, Jacobian supports feedforward neural networks and has partial support for convolutional neural networks.

## Usage

Initializing and training a neural network with Jacobian takes just 8 lines of code!
```python
import jacobian as jcb
net = jcb.Network("./data_banknote_authentication.txt", 10, 0.0155, 0.03, jcb.L2, 1, 0.9)
net.add_layer(4, jcb.activations.linear, jcb.activations.linear_deriv)
net.add_layer(5, jcb.activations.sigmoid, jcb.activations.sigmoid_deriv)
net.add_layer(2, jcb.activations.linear, jcb.activations.linear_deriv)
# Optional: net.init_optimizer(jcb.optimizers.momentum(0.1))
# Optional: net.init_decay(jcb.decays.exponential(1, 0.5))
net.initialize()
for i in range(50):
  net.train()
```
## Examples

See `example.cpp` for an example of using Jacobian from C++, and `example.py` for an example of using Jacobian from Python.

## Building

### Dependencies
Eigen 3 is the only dependency, although building the python library requires `pybind11`.

<!-- ### Main Steps
Note: This is all ideally the process for manual building, but it's so confusing that I'm not sure. You're better off manually copying the .so file!
1. Install the C++ library Eigen. Ensure the that the path eigen3/Eigen/Dense is within one of your include directories.
2. Install the C++ and python ends of the pybind11 library.
3. Run `make` with the configuration of your choosing.
4. Run `python setup.py bdist_wheel`
5. `cd` into `dist` and run `python3 -m pip install --upgrade Jacobian-1.0-cp37-cp37m-macosx_10_13_x86_64.whl`
6. Be unhappy when it doesn't work out and resort to just copying .so files. -->

### Building with CMake

**Don't forget to delete `CMakeCache.txt` after each compilation if you plan on switching things up!**

There are two target languages, five main build configurations, and a number of toggleable build 'attributes'. The preferred target language can be specified by setting a CMake  variable from the command-line: `-DCXX=ON` or `-DPYTHON=ON`.

The five main configurations correspond to differing levels of optimization.

- `cmake .`: No compiler optimizations.
- `cmake . -DFAST=ON`: Enables the O3 optimization layer in the compiler.
- `cmake . -DFASTER=ON`: Enables O3 as well as extra individual flags.
- `cmake . -DTRADEOFFS=ON`: All previous optimizations as well as ones that sacrifice precision.
- `cmake . -DRECKLESS=ON`: Like `TRADEOFFS`, but defines the RECKLESS macro (and NDEBUG) which skips all checks within the code.

One you've selected a main optimization level, extra configurations can be passed in.

- `-DDEBUG=ON` enables debugging features in the compiler (and shows warnings).

A sample build process would look like this:

```sh
rm CMakeCache.txt && cmake . -DPYTHON=ON -DFASTER=ON -DDEBUG=ON && make
```
