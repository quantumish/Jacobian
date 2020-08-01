//
//  example.cpp
//  Jacobian
//
//  Created by David Freifeld
//  Copyright © 2020 David Freifeld. All rights reserved.
//

#include "./src/bpnn.hpp"
#include "./src/utils.hpp"
#include "unistd.h"
#include <ctime>

double bench(int batch_sz)
{
  auto start = std::chrono::high_resolution_clock::now();
  Network net ("./data_banknote_authentication.txt", batch_sz, 0.0155, 0.03, 2, 0.01, 0.9);
  net.add_layer(4, "linear");
  net.add_layer(5, "relu");
  net.add_layer(2, "linear");
  net.init_optimizer("momentum", 0.9);
  net.initialize();
  std::vector<float> vals;
  for (int i = 0; i < 50; i++) {
    net.train();
    vals.push_back(net.get_val_acc());
  }
  std::cout <<  "[";
  for (int i = 0; i < vals.size(); i++) {
    if (i == 49) std::cout << vals[i];
    else std::cout << vals[i] << ", ";
  }
  std::cout <<  "]";
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / pow(10,9);
}

int main()
{
  std::cout << bench(16) << "\n";
  //  bench(50);
  //  bench(50);
  //  bench(50);
  //  bench(50);
}
