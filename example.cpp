#include "./src/bpnn.hpp"
#include "./src/utils.hpp"
#include "unistd.h"
#include <ctime>

double bench(int batch_sz)
{
  auto start = std::chrono::high_resolution_clock::now();
  Network net ("./data_banknote_authentication.txt", batch_sz, 0.0155, 0.03, 0, 0.9);
  net.add_layer(4, "linear");
  net.add_layer(5, "relu");
  net.add_layer(2, "sigmoid");
  net.init_decay("step", 1, 2);
  net.initialize();
  // checks(net);
  for (int i = 0; i < 5; i++) {
    net.next_batch();
    net.feedforward();
    net.list_net();
    net.backpropagate();
    std::cout << net.cost() << " " << net.accuracy() << "\n";
  }
  // for (int i = 0; i < 1; i++) {
  //   net.train();
  // }
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
