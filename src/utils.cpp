//
//  utils.cpp
//  Jacobian
//
//  Created by David Freifeld
//  Copyright © 2020 David Freifeld. All rights reserved.
//

#include <iostream>
#include <fstream>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <cstdio>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <Eigen/Dense>
#include <Eigen/MatrixFunctions>

// A bunch of hardcoded activation functions. Avoids much of the slowness of custom functions.
// Although the std::function makes it not the fastest way, the functionality is worth it.
// Yes, these functions may be a frustrating to read but they're just equations and I want to conserve space.

float sigmoid(float x) {return 1.0/(1+exp(-x));}
float sigmoid_deriv(float x) {return 1.0/(1+exp(-x)) * (1 - 1.0/(1+exp(-x)));}

float linear(float x) {return x;}
float linear_deriv(float x) {return 1;}

float lecun_tanh(float x) {return 1.7159 * tanh((2.0/3) * x);}
float lecun_tanh_deriv(float x) {return 1.14393 * pow(1.0/cosh(2.0/3 * x),2);}

float inverse_logit(float x) {return (exp(x)/(exp(x)+1));}
float inverse_logit_deriv(float x) {return (exp(x)/pow(exp(x)+1, 2));}

float softplus(float x) {return log(1+exp(x));}
float softplus_deriv(float x) {return exp(x)/(exp(x)+1);}

float cloglog(float x) {return 1-exp(-exp(x));}
float cloglog_deriv(float x) {return exp(x-exp(x));}

float step(float x)
{
    if (x > 0) return 1;
    else return 0;
}
float step_deriv(float x) {return 0;}

float bipolar(float x)
{
    if (x > 0) return 1;
    else if (x == 0) return 0;
    else return -1;
}
float bipolar_deriv(float x) {return 0;}

float bipolar_sigmoid(float x) {return (1-exp(-x))/(1+exp(-x));}
float bipolar_sigmoid_deriv(float x) {return (2*exp(x))/(pow(exp(x)+1,2));}

float hard_tanh(float x) {return fmax(-1, fmin(1,x));}
float hard_tanh_deriv(float x)
{
    if (-1 < x && x < 1) return 1;
    else return 0;
}

float leaky_relu(float x)
{
    if (x > 0) return x;
    else return 0.01 * x;
}

float leaky_relu_deriv(float x)
{
    if (x > 0) return 1;
    else return 0.01;
}

std::function<float(float)> rectifier(float (*activation)(float))
{
    auto rectified = [activation](float x) -> float
    { 
        if (x > 0) return (*activation)(x);
        else return 0; 
    };
    return rectified;
}

// Intel intrinsics for the win!
Eigen::MatrixXf avx_product(Eigen::MatrixXf a, Eigen::MatrixXf b)
{
#ifndef RECKLESS
    assert(a.rows() == b.rows() && a.cols() == b.rows());
#endif
    int size = ((a.rows() * a.cols()) + 7) & (-8);
    for (int i = 0; i < (size-8)/8; i++) {
        _mm256_store_ps(a.data()+i*8, _mm256_mul_ps(_mm256_load_ps(a.data()+i*8), _mm256_load_ps(b.data()+i*8)));
    }
    return a;
}

Eigen::MatrixXf avx_exp(Eigen::MatrixXf m)
{
    int size = ((m.rows() * m.cols()) + 7) & (-8);
    for (int i = 0; i < (size-8)/8; i++) {
        _mm256_store_ps(m.data()+i*8, _mm256_exp_ps(_mm256_load_ps(m.data()+i*8)));
    }
    return m;
}

Eigen::MatrixXf avx_cdiv(Eigen::MatrixXf m, float denom)
{
    int size = ((m.rows() * m.cols()) + 7) & (-8);
    for (int i = 0; i < (size-8)/8; i++) {
        __m256 denom_vec = _mm256_broadcast_ss(&denom);
        _mm256_store_ps(m.data()+i*8, _mm256_div_ps(_mm256_load_ps(m.data()+i*8), denom_vec));
    }
    return m;
}

Eigen::MatrixXf avx_log(Eigen::MatrixXf m)
{
    int size = ((m.rows() * m.cols()) + 7) & (-8);
    for (int i = 0; i < (size-8)/8; i++) {
        _mm256_store_ps(m.data()+i*8, _mm256_log_ps(_mm256_load_ps(m.data()+i*8)));
    }
    return m;
}


Eigen::MatrixXf avx_pow(Eigen::MatrixXf m, float exponent)
{
    int size = ((m.rows() * m.cols()) + 7) & (-8); 
    for (int i = 0; i < (size-8)/8; i++) {
        __m256 exponent_vec = _mm256_broadcast_ss(&exponent);
        _mm256_store_ps(m.data()+i*8, _mm256_pow_ps(_mm256_load_ps(m.data()+i*8), exponent_vec));
    }
    return m;
}


Eigen::MatrixXf avx_tanh(Eigen::MatrixXf m)
{
    int size = ((m.rows() * m.cols()) + 7) & (-8);
    for (int i = 0; i < (size-8)/8; i++) {
        _mm256_store_ps(m.data()+i*8, _mm256_tanh_ps(_mm256_load_ps(m.data()+i*8)));
    }
    return m;
}

Eigen::MatrixXf avx_cosh(Eigen::MatrixXf m)
{
    int size = ((m.rows() * m.cols()) + 7) & (-8);
    for (int i = 0; i < (size-8)/8; i++) {
        _mm256_store_ps(m.data()+i*8, _mm256_cosh_ps(_mm256_load_ps(m.data()+i*8)));
    }
    return m;
}
