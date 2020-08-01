
  <!-- readme.md -->
  <!-- Jacobian -->

  <!-- Created by David Freifeld -->
  <!-- Copyright © 2020 David Freifeld. All rights reserved. -->

![Banner](./pictures/banner.png)

## About
Jacobian is a work-in-progress machine learning library written in C++ designed to run as fast as possible while still being simple to use. Jacobian is accessible via Python and enables you to write models that train faster in the same amount of code. As of now, Jacobian supports feedforward neural networks and has partial support for convolutional neural networks. ***Note that as Jacobian is a work-in-progress, the latest commit is and will be largely unstable until convolutional networks and multiclass classification are fully implemented.***

## Benchmark Info

One of the tradeoffs of Jacobian is that as of now it doesn't train nearly as close to perfection as other available libraries and doesn't maximize accuracy as much (with the benefit being the added speed. Here's a graph of Jacobian's model metrics over epochs on a simple task (banknote dataset with batch size 16) as compared to other libraries.

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

First, call the Network constructor (from henceforth all functions will be presented in their C++ form). This is where many of the hyperparameters are defined, and you'll need to pass in the path to your data, the desired batch size, learning rate, bias learning rate (which should usually be smaller), regularization strength, and train-val ratio.
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
In the `/scripts` directory there is a example of a neural network being used in conjuction with Weights & Biases, allowing for effective hyperparameter searches and accuracy reporting.

## Installing

### Main Steps
Note: This is all ideally the process, but it's so confusing that I'm not sure. You're better off manually copying the .so file!
1. Install the C++ library Eigen.
2. Install the C++ and python ends of the pybind11 library.
3. Run `make` with the configuration of your choosing.
4. Run `python setup.py bdist_wheel`
5. `cd` into `dist` and run `python3 -m pip install --upgrade Jacobian-1.0-cp37-cp37m-macosx_10_13_x86_64.whl`
6. Be unhappy when it doesn't work out and resort to just copying .so files.

### Build Configurations (Building from Source)

There are 5 build configurations, each one prioritizing program speed more than the last. *Warning!* The higher levels (>fast) are as of now unstable as changes to the code have caused it to not play nicely with MKL or icpc.

#### Level 1: `make`
Simply builds the project with no optimization at all. Use this if you don't want to wait long for the library to compile and don't care too much about speed in the moment.

#### Level 2: `make fast`
  - Builds the project with the O3 optimization setting, essentially the highest optimization configuration without manually passing optimization flags (Ofast can prove slower).

Use this if you want some speed but shorter build commands and compile times.

#### Level 3: `make faster`
Enables a whole slew of extra optimizations, some of which include:
  - Building the project with O3.
  - Optimizing for native architecture.
  - Instructing compiler to fetch data for CPU cache earlier.
  - Unrolling loops.
  - Links with Intel's Math Kernel Library (make sure you have this!). Provides an extra boost to speed.

  Use this if you care a lot about speed but are not willing to sacrifice anything but compile time for it.

#### Level 4: `make tradeoffs`
Builds the project with O3, specific optimization flags, and *more* specific optimization flags that sacrifice things like portability and precision as well as makes assumptions to increase speed even further. Use this if you care about speed more than precision and are willing to make some tradeoffs (and also don't mind longer compile times).

Some of the new compiler optimizations include:
   - Using an optimized version of `calloc`.
   - Using Intel-specific optimizations.
   - Allowing low-precision alternatives to operations like sqrt and division.
   - Enabling the `-ffast-math` flag.
   - Approximating more complicated functions.

In the future this setting may try to parallelize operations (even if the network is already parallelized with MapReduce).

#### Level 5: `make reckless`
This option is not implemented as of now.

Planned features include:
    - Adding all the aforementioned compiler options as well as compiler options that are potentially unsafe.
    - Defining the `RECKLESS` macro which will skip anything that is not absolutely necessary in the code (with preprocessor statements like `#ifndef`). Will skip all sort of testing and checking before training.

## The Future

Jacobian is actively in development and the following are things that are planned for the nearish future:
- More architectures such as Convolutional Neural Networks and RNN-like architectures (LSTM, GRU...).
- Moving towards a more proper release by emphasizing usability.
- Further increasing speedups from a conceptual perspective with better algorithms, a implementation perspective with optimized code, and a low-level perspective with hardware optimizations + more compiler work.
- More advanced capabilities such as the inclusion of gradient descent optimizations.
- Parallelization, data-oriented design, and more ways of increasing usability+speed.
