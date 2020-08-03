//
//  example.cpp
//  Jacobian
//
//  Created by David Freifeld
//

#include "./src/bpnn.hpp"
#include "./src/utils.hpp"
#include "unistd.h"
#include <ctime>

double bench(int batch_sz)
{
  auto start = std::chrono::high_resolution_clock::now();
  Network net ("./data_banknote_authentication.txt", batch_sz, 0.0155, 0.03, 2, 0, 0.9);
  net.add_layer(4, "linear");
  net.add_layer(5, "relu");
  net.add_layer(2, "linear");
  net.initialize();
  for (int i = 0; i < 50; i++) {
    net.train();
  }
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / pow(10,9);
}

int main()
{
  sleep(40);
  bench(16);
}
