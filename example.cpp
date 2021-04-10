//
//  example.cpp
//  Jacobian
//
//  Created by David Freifeld
//

// #include <indicators/cursor_control.hpp>
// #include <indicators/progress_bar.hpp>
// #include <indicators/block_progress_bar.hpp>
// using namespace indicators;

#include "src/bpnn.hpp"
#include "src/utils.hpp"
#include "unistd.h"
#include <ctime>
#include <chrono>

double bench(int batch_sz, int epochs)
{
	auto start = std::chrono::high_resolution_clock::now();
	Jacobian::Network net ("./data_banknote_authentication.txt", batch_sz, 0.0155, 0.03, Jacobian::Regularization::L2, 0, 0.9);
	net.add_layer(4, Jacobian::activations::linear, Jacobian::activations::linear_deriv);
	net.add_layer(5, Jacobian::activations::lecun_tanh, Jacobian::activations::lecun_tanh_deriv);
	net.add_layer(2, Jacobian::activations::linear, Jacobian::activations::linear_deriv);
	net.init_optimizer(Jacobian::optimizers::momentum(0.1));
	net.initialize();
	for (int i = 0; i < epochs; i++) {
		net.train();
	}
	auto end = std::chrono::high_resolution_clock::now();
	return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / pow(10,9);
}

int main(int argc, char** argv)
{
	if (argc < 2) {
		std::cout << "Invalid command! Either pass a special option or pass two integers - batch_size and epochs (in that order)."  << "\n";
		exit(1);
	}
	else {
		sleep(strtol(argv[3], NULL, 10));
		std::cout << bench(strtol(argv[1], NULL, 10), strtol(argv[2], NULL, 10)) << "\n";
	}
}
