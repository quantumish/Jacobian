#include "bpnn.hpp"
#include "utils.hpp"
#include "unistd.h"
#include <ctime>

double bench(int batch_sz)
{
  auto start = std::chrono::high_resolution_clock::now();
  Network net ("./data_banknote_authentication.txt", batch_sz, 0.0155, 0.03, 0.9);
  net.add_layer(4, "linear");
  net.add_layer(5, "relu");
  net.add_layer(1, "resig");
  net.initialize();
  //  net.begin();
  for (int i = 0; i < 50; i++) {
    net.train();
  }
  auto end = std::chrono::high_resolution_clock::now();
  //  net.list_net();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / pow(10,9);
}

int main()
{
  std::cout << bench(50) << "\n";
  //  bench(50);
  //  bench(50);
  //  bench(50);
  //  bench(50);
}
