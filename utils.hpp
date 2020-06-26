#ifndef UTILS_H
#define UTILS_H

#include <functional>

double sigmoid(double x);
double sigmoid_deriv(double x);
double linear(double x);
double linear_deriv(double x);
std::function<double(double)> rectifier(double (*activation)(double));

#endif /* MODULE_H */
