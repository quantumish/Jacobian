#ifndef UTILS_H
#define UTILS_H

#include <functional>

double sigmoid(double x);
double sigmoid_deriv(double x);
double linear(double x);
double linear_deriv(double x);
double step(double x);
double step_deriv(double x);
double bipolar(double x);
double bipolar_deriv(double x);
double logit(double x);
double logit_deriv(double x);
double lecun_tanh(double x);
double lecun_tanh_deriv(double x);
std::function<double(double)> rectifier(double (*activation)(double));

#endif /* MODULE_H */
