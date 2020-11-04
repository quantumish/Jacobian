
  <!-- readme.md -->
  <!-- Jacobian -->

  <!-- Created by David Freifeld -->

![Banner](./pictures/banner.png)

<img align="center" height=35 width=300 src="https://forthebadge.com/images/badges/powered-by-electricity.svg">

<img align="center" height=35 width=300 src="https://forthebadge.com/images/badges/works-on-my-machine.svg">

<img align="center" height=35 width=290 src="https://forthebadge.com/images/badges/60-percent-of-the-time-works-every-time.svg">

## About
Jacobian is a work-in-progress machine learning library written in C++ designed to run as fast as possible while still being simple to use. Jacobian is accessible via Python and enables you to write models that train faster with the same amount of code. As of now, Jacobian supports feedforward neural networks and has partial support for convolutional neural networks. ***Note that as Jacobian is a work-in-progress, the latest commit is and will be largely unstable until this README specifies otherwise.***

## Benchmarks

One of the tradeoffs of Jacobian is that as of now it doesn't train nearly as close to perfection as other available libraries nor does it maximize accuracy as much (with the benefit being the added speed). Here's a graph of Jacobian's model metrics over epochs on a simple task (banknote dataset with batch size 16) as compared to other libraries.

![Loss vs. Epochs](./pictures/metrics_updated.png)

Here's a runtime comparison for a simple example task (same as before) between Jacobian and some other popular ML libraries:

![Runtime Comparison](./pictures/updated_runtime.png)

**Coming soon:** A more detailed and current rundown of the speed of Jacobian vs popular machine learning libraries for Python (and eventually comparisons to C++ libraries as well) as well as a handy and flexible Python script for creating benchmark graphs on the fly.


## Usage

Initializing and training a neural network with Jacobian takes just 8 lines of code!
```python
import mrbpnn
net = mrbpnn.Network("../data_banknote_authentication.txt", 10, 0.01, 0.001, 0.5, 0.75)
net.add_layer(4, "linear")
net.add_layer(10, "lecun_tanh")
net.add_layer(1, "linear")
net.initialize()
for i in range(50):sr
  net.train()
```

### Rundown
What's happening in those seven lines?

First, call the Network constructor (from henceforth all functions will be presented in their C++ form for clarity). This is where many of the hyperparameters are defined, and you'll need to pass in the path to your data, the desired batch size, learning rate, bias learning rate (which should usually be smaller), regularization strength, and train-val ratio.
```c++
Network(char* path, int batch_sz, float learn_rate, float bias_rate, float l, float ratio);
```

Then add your layers one by one, similar to Keras. You'll need to specify how many neurons there are in the layer and the layer's activation.
```c++
void add_layer(int nodes, char* activation);
```
Optionally, one can overwrite the activation of a layer with their own function:
```c++
void set_activation(int index, std::function<float(float)> custom, std::function<float(float)> custom_deriv);
```
At this point could also specify a learning rate decay function as well by calling `init_decay`.
```c++
void init_decay(char* type, float a_0, float k);
```
Next, call `initialize()` to initialize the network's weights.
Finally, train your network for one epoch with `train()`. Training one epoch at a time allows you control over the accuracy reporting (with functions like `get_cost()`, `get_accuracy()`, `get_val_cost()`, and `get_val_accuracy()`) and also allows effective use of services like W&B.

### Examples
In the `/scripts` directory there is an example of a neural network being used in conjuction with Weights & Biases, allowing for effective hyperparameter searches and accuracy reporting.

## Building

### Main Steps
Note: This is all ideally the process for manual building, but it's so confusing that I'm not sure. You're better off manually copying the .so file!
1. Install the C++ library Eigen.
2. Install the C++ and python ends of the pybind11 library.
3. Run `make` with the configuration of your choosing.
4. Run `python setup.py bdist_wheel`
5. `cd` into `dist` and run `python3 -m pip install --upgrade Jacobian-1.0-cp37-cp37m-macosx_10_13_x86_64.whl`
6. Be unhappy when it doesn't work out and resort to just copying .so files.

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
- `-DAVX=ON` enables explicit AVX function calls within the code. **Warning: this will not work without the Intel C++ Compiler!**

A sample build process would look like this: 

```
cmake . -DCXX=ON -DFASTER=ON -DAVX=ON -DDEBUG=ON
make
```
