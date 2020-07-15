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
  Network net ("./data_banknote_authentication.txt", batch_sz, 0.1, 0.03, 0, 0.9);
  net.add_layer(4, "linear");
  net.add_layer(6, "lecun_tanh");
  net.add_layer(2, "linear");
  //  net.init_decay("step", 1, 2);
  net.initialize();
  //  checks(net);
  // for (int i = 0; i < 10; i++) {
  //   net.next_batch();
  //   net.feedforward();
  //   net.list_net();
  //   net.backpropagate();
  //   std::cout << net.cost() << " " << net.accuracy() << "\n";
  // }
  for (int i = 0; i < 50; i++) {
    net.train();
    //    net.list_net();
  }
  //  std::cout << *net.layers[net.length-1].contents << "\n\n";
  //  std::cout << *net.labels << "\n";
  auto end = std::chrono::high_resolution_clock::now();
  //net.list_net();
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
