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

#include "utils.hpp"

namespace Jacobian {
namespace activations {
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
	auto rectified = [activation](float x) -> float {
		if (x > 0)
			return (*activation)(x);
		else
			return 0;
	};
	return rectified;
}
} // namespace activations

namespace optimizers {
std::function<void(Layer&, Eigen::MatrixXf, float)> momentum(float beta) {
	return [beta](Layer& layer, const Eigen::MatrixXf delta, const float learning_rate) {
	  layer.weights -= (beta * layer.m) + (learning_rate * delta);
	  layer.m = (learning_rate * delta);
	};
}

std::function<void(Layer&, Eigen::MatrixXf, float)> demon(float beta, int max_ep) {
	float beta_init = beta;
	float prev_epoch = -1;
	float epochs = 0;
	return [max_ep, epochs, beta_init, beta](Layer& layer, const Eigen::MatrixXf delta, const float learning_rate) mutable {
		beta = beta_init * (1-(epochs/max_ep)) / ((beta_init * (1-(epochs/max_ep))) + (1-beta_init));
		layer.weights -= (beta * layer.m) + (learning_rate * delta);
		layer.m = (learning_rate * delta);
		epochs++;
	};
}

std::function<void(Layer&, Eigen::MatrixXf, float)> adam(float beta1, float beta2, float epsilon) {
	return [beta1, beta2, epsilon](Layer& layer, const Eigen::MatrixXf delta, const float learning_rate) {
		layer.m = (beta1 * layer.m) + ((1-beta1)*delta);
		layer.v = (beta2 * layer.v) + (1-beta2)*(delta.cwiseProduct(delta));
		layer.weights -= learning_rate *
			((layer.v.cwiseSqrt()).array()+epsilon).pow(-1).cwiseProduct(layer.m.array()).matrix();
	};
}

std::function<void(Layer&, Eigen::MatrixXf, float)> adamax(float beta1, float beta2, float epsilon) {
	return [beta1, beta2, epsilon](Layer &layer,
					   const Eigen::MatrixXf delta,
					   const float learning_rate) {
		layer.m = (beta1 * layer.m) + ((1 - beta1) * delta);
		if ((beta2 * layer.v).sum() > delta.array().abs().sum())
			layer.v = (beta2 * layer.v);
		else
			layer.v = delta.array().abs().matrix();
		layer.weights -=
			learning_rate *
			(layer.v.array().pow(-1).cwiseProduct(layer.m.array()))
				.matrix();
	};
}
}
}
