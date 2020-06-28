#include "bpnn.hpp"
#include "utils.hpp"
#include "unistd.h"
#include <ctime>

int main()
{
  sleep(30);
  Network net ("./data_banknote_authentication.txt", 10, 0.01, 0.001);
  net.add_layer(4, "linear");
  net.add_layer(5, "lecun_tanh");
  net.add_layer(1, "resig");
  net.initialize();
  //  net.list_net();
  net.train(500);
  net.list_net();
  //printf("%i\n", wc("./data_banknote_authentication.txt"));

  // double x = 0.4235;
  // auto bench_start = std::chrono::high_resolution_clock::now();
  // x = tanh(x);
  // printf("%lf", x);
  // auto tanh_end = std::chrono::high_resolution_clock::now();
  // cosh(x);
  // auto cosh_end = std::chrono::high_resolution_clock::now();
  // exp(x);
  // auto exp_end = std::chrono::high_resolution_clock::now();
  // log(x);
  // auto log_end = std::chrono::high_resolution_clock::now();
  // x = x - (1/3 * pow(x, 3)) + (2/15 * pow(x, 5)) - (17/315 * pow(x, 7));
  // printf("%lf", x);
  // auto pow_end = std::chrono::high_resolution_clock::now();
  // std::cout << " TANH " << std::chrono::duration_cast<std::chrono::nanoseconds>(tanh_end - bench_start).count() << " COSH " << std::chrono::duration_cast<std::chrono::nanoseconds>(cosh_end - tanh_end).count() << " EXP " << std::chrono::duration_cast<std::chrono::nanoseconds>(exp_end - cosh_end).count() << " BETTER TANH? " << std::chrono::duration_cast<std::chrono::nanoseconds>(log_end - exp_end).count() << " POW " << std::chrono::duration_cast<std::chrono::nanoseconds>(pow_end - log_end).count() << "\n";
}
