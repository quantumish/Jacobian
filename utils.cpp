#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <cstdio>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <Eigen/Dense>

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

Eigen::MatrixXd strassen_mul(Eigen::MatrixXd x, Eigen::MatrixXd y)
{
  int apower;
  int bpower;
  for (;a
  //Eigen::MatrixXd a ()
  int block_len = a.rows()/2;
  Eigen::MatrixXd result (a.rows(), a.cols());

  Eigen::MatrixXd m1 = ((a.block(0,0, block_len, block_len)) + a.block(a.rows()-block_len,a.cols()-block_len, block_len, block_len)) * (b.block(0,0, block_len, block_len) + b.block(b.rows()-block_len,b.cols()-block_len, block_len, block_len));
  Eigen::MatrixXd m2 = (a.block(a.rows()-block_len, 0, block_len, block_len) + a.block(a.rows()-block_len,a.cols()-block_len, block_len, block_len)) * (b.block(0,0, block_len, block_len));
  Eigen::MatrixXd m3 = a.block(0,0, block_len, block_len) * (b.block(0,b.cols()-block_len, block_len, block_len) - b.block(b.rows()-block_len,b.cols()-block_len, block_len, block_len));
  Eigen::MatrixXd m4 = a.block(a.rows()-block_len,a.cols()-block_len, block_len, block_len) * (b.block(b.rows()-block_len,0, block_len, block_len) - b.block(0,0, block_len, block_len));
  Eigen::MatrixXd m5 = (a.block(0, 0, block_len, block_len) + a.block(0,a.cols()-block_len, block_len, block_len)) * (b.block(b.rows()-block_len,b.cols()-block_len, block_len, block_len));
  Eigen::MatrixXd m6 = (a.block(a.rows()-block_len,0, block_len, block_len) - a.block(0,0, block_len, block_len)) * (b.block(0,0, block_len, block_len) + b.block(0,b.cols()-block_len, block_len, block_len));
  Eigen::MatrixXd m7 = (a.block(0,a.cols()-block_len, block_len, block_len) - a.block(a.rows()-block_len,a.cols()-block_len, block_len, block_len)) * (b.block(a.rows()-block_len,0, block_len, block_len) + b.block(b.rows()-block_len,b.cols()-block_len, block_len, block_len));
  
  result.block(0,0, block_len, block_len) =  m1 + m4 - m5 + m7;
  result.block(0,result.cols()-block_len, block_len, block_len) =  m3 + m5;
  result.block(result.rows()-block_len,0, block_len, block_len) =  m2 + m4;
  result.block(result.rows()-block_len,result.cols()-block_len, block_len, block_len) =  m1 -m2 + m3 + m6;

  return result;
}
