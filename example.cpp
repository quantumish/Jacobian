#include "bpnn.hpp"
#include "utils.hpp"

int main()
{
  Network net ("./data_banknote_authentication.txt", 10, 0.0001, 0.000001);
  net.add_layer(4, "linear");
  //net.add_layer(3, "resig");
  net.add_layer(5, "sigmoid");
  net.add_layer(1, "resig");
  net.initialize();
  net.list_net();
  net.train(500);
  //  net.list_net();
  //char line[1024];
  //net.stream->getline(line, 1024);
  //std::cout << line << "\n";
}
