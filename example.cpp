#include "bpnn.hpp"
#include "utils.hpp"

int main()
{
  Network net ("./extra.txt", 4, 1, 1, 5, 10, 1);
  net.set_activation(0, "linear");
  net.set_activation(1, "sigmoid");
  net.set_activation(2, "sigmoid");
  net.set_activation(3, "resig");
  //  net.list_net();
  net.train(1);
  //  net.list_net();
  //char line[1024];
  //net.stream->getline(line, 1024);
  //std::cout << line << "\n";
}
