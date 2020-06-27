#ifndef UTILS_H
#define UTILS_H

#include <functional>

// A zoo of activation functions.
double sigmoid(double x);
double sigmoid_deriv(double x);
double linear(double x);
double linear_deriv(double x);
double step(double x);
double step_deriv(double x);
double bipolar(double x);
double bipolar_deriv(double x);
double mytanh(double x);
double tanh_deriv(double x);
double lecun_tanh(double x);
double lecun_tanh_deriv(double x);
double cloglog(double x);
double cloglog_deriv(double x);
double softplus(double x);
double softplus_deriv(double x);
double inverse_logit(double x);
double inverse_logit_deriv(double x);
std::function<double(double)> rectifier(double (*activation)(double));

// Experimental fast file reading functions
uintmax_t wc(char const *fname);
#endif /* MODULE_H */
