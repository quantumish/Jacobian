#include "bpnn.hpp"
#include "utils.hpp"
#include "unistd.h"
#include <ctime>

double bench(int batch_sz)
{
  auto start = std::chrono::high_resolution_clock::now();
  Network net ("./data_banknote_authentication.txt", batch_sz, 0.0155, 0.03);
  net.add_layer(4, "linear");
  net.add_layer(5, "relu");
  net.add_layer(1, "resig");
  net.initialize();
  net.train(50);
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / pow(10,9);
}

int main()
{
  double x[1340];
  for(int i = 0; i < 3; i++) {
    x[i] = bench(i);
  }
  printf("[");
  for(int i = 0; i < 3; i++) {
    printf("%d, ", x[i]);
  }
  printf("]");
}
