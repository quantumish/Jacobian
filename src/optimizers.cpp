//
//  optimizers.cpp
//  Jacobian
//
//  Created by David Freifeld
//


std::function<void(Layer&, Eigen::MatrixXf, float)> optimizers::momentum(float beta) {
	return [beta](Layer& layer, const Eigen::MatrixXf delta, const float learning_rate) {
	  layer.weights -= (beta * layer.m) + (learning_rate * delta);
	  layer.m = (learning_rate * delta);
	};
}

std::function<void(Layer&, Eigen::MatrixXf, float)> optimizers::demon(float beta, int max_ep) {
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

std::function<void(Layer&, Eigen::MatrixXf, float)> optimizers::adam(float beta1, float beta2, float epsilon) {
	return [beta1, beta2, epsilon](Layer& layer, const Eigen::MatrixXf delta, const float learning_rate) {
		layer.m = (beta1 * layer.m) + ((1-beta1)*delta);
		layer.v = (beta2 * layer.v) + (1-beta2)*(delta.cwiseProduct(delta));
		layer.weights -= learning_rate *
			((layer.v.cwiseSqrt()).array()+epsilon).pow(-1).cwiseProduct(layer.m.array()).matrix();
	};
}

std::function<void(Layer&, Eigen::MatrixXf, float)> optimizers::adamax(float beta1, float beta2, float epsilon) {
	return [beta1, beta2, epsilon](Layer& layer, const Eigen::MatrixXf delta, const float learning_rate) {
		layer.m = (beta1 * layer.m) + ((1-beta1)*delta);
		if ((beta2 * layer.v).sum() > delta.array().abs().sum()) layer.v = (beta2 * layer.v);
		else layer.v = delta.array().abs().matrix();
		layer.weights -= learning_rate *
			(layer.v.array().pow(-1).cwiseProduct(layer.m.array())).matrix();
	};
}

void Network::init_optimizer(std::function<void(Layer&, Eigen::MatrixXf, float)> f)
{
	update = f;
}
