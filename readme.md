
  <!-- readme.md -->
  <!-- Jacobian -->

  <!-- Created by David Freifeld -->

## About

This is the branch with now-deprecated AVX calls for matrix operations intended to speed up softmax and coefficient wise multipications. It is not faster than the regular code compiled with -O3 (aka -DFAST or above). 

## Building

**Don't forget to delete `CMakeCache.txt` after each compilation if you plan on switching things up!**

This branch provides an extra configuration flag to be passed in alongside normal build parameters.

- `-DAVX=ON` enables explicit AVX function calls within the code. **Warning: this will not work without the Intel C++ Compiler as it relies on a proprietary library!**

A sample build process would look like this: 

```
cmake . -DCXX=ON -DFASTER=ON -DAVX=ON -DDEBUG=ON
make
```
