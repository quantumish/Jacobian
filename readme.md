
  <!-- readme.md -->
  <!-- Jacobian -->

  <!-- Created by David Freifeld -->

## About
This the branch with support for Parametric ReLU (a version of leaky ReLU where the coefficient for x<0 is trained with backprop) based off of [this paper](https://arxiv.org/abs/1502.01852). 

## Usage
Just use the following function to add a PreLU layer to your network;

```c++
void add_prelu_layer(int nodes, float a);
```
where `nodes` is the number of neurons in the layer, and `a` is the initial coefficient.
