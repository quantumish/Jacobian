#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <cstdio>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>

// A bunch of hardcoded activation functions. Avoids much of the slowness of custom functions.
// Although the std::function makes it not the fastest way, the functionality is worth it.
// Yes, these functions may be a frustrating to read but they're just equations and I want to conserve space.

//double tanhapprox(double x) {return x - (1/3 * pow(x, 3)) + (2/15 * pow(x, 5)) - (17/315 * pow(x, 7));}

double sigmoid(double x) {return 1.0/(1+exp(-x));}
double sigmoid_deriv(double x) {return 1.0/(1+exp(-x)) * (1 - 1.0/(1+exp(x)));}

double linear(double x) {return x;}
double linear_deriv(double x) {return 1;}

double lecun_tanh(double x) {return 1.7159 * tanh((2.0/3) * x);}
double lecun_tanh_deriv(double x) {return 1.14393 * pow(1.0/cosh(2.0/3 * x),2);}

double inverse_logit(double x) {return (exp(x)/(exp(x)+1));}
double inverse_logit_deriv(double x) {return (exp(x)/pow(exp(x)+1, 2));}

double softplus(double x) {return log(1+exp(x));}
double softplus_deriv(double x) {return exp(x)/(exp(x)+1);}

double cloglog(double x) {return 1-exp(-exp(x));}
double cloglog_deriv(double x) {return exp(x-exp(x));}

double step(double x)
{
  if (x > 0) return 1;
  else return 0;
}
double step_deriv(double x) {return 0;}

double bipolar(double x)
{
  if (x > 0) return 1;
  else return -1;
}
double bipolar_deriv(double x) {return 0;}

std::function<double(double)> rectifier(double (*activation)(double))
{
  auto rectified = [activation](double x) -> double
  { 
    if (x > 0) return (*activation)(x);
    else return 0; 
  };
  return rectified;
}
