//
//  utils.hpp
//  Jacobian
//
//  Created by David Freifeld
//

#ifndef UTILS_H
#define UTILS_H

#include <functional>
#include <Eigen/Dense>
#include "bpnn.hpp"

namespace Jacobian {
namespace activations {
float sigmoid(float x);
float sigmoid_deriv(float x);
float linear(float x);
float linear_deriv(float x);
float step(float x);
float step_deriv(float x);
float bipolar(float x);
float bipolar_deriv(float x);
float mytanh(float x);
float tanh_deriv(float x);
float lecun_tanh(float x);
float lecun_tanh_deriv(float x);
float cloglog(float x);
float cloglog_deriv(float x);
float softplus(float x);
float softplus_deriv(float x);
float inverse_logit(float x);
float inverse_logit_deriv(float x);
float hard_tanh(float x);
float hard_tanh_deriv(float x);
float bipolar_sigmoid(float x);
float bipolar_sigmoid_deriv(float x);
float leaky_relu(float x);
float leaky_relu_deriv(float x);
std::function<float(float)> rectifier(float (*activation)(float));
}

namespace optimizers {
std::function<void(Layer&, Eigen::MatrixXf, float)> momentum(float beta);
std::function<void(Layer&, Eigen::MatrixXf, float)> demon(float beta_init, int max_ep);
std::function<void(Layer&, Eigen::MatrixXf, float)> adam(float beta1, float beta2, float epsilon);
std::function<void(Layer&, Eigen::MatrixXf, float)> adamax(float beta1, float beta2, float epsilon);
}

namespace decays {
std::function<void(float &)> step(float a_0, float k);
std::function<void(float &)> exponential(float a_0, float k);
std::function<void(float &)> fractional(float a_0, float k);
std::function<void(float &)> linear(int max_ep);
}

Eigen::MatrixXf strassen_mul(Eigen::MatrixXf a, Eigen::MatrixXf b);

}

#endif /* MODULE_H */
