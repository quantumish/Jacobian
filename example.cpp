#include "bpnn.hpp"
#include "utils.hpp"

int main()
{
  Network net ("./data_banknote_authentication.txt", 10, 0.01, 0.001);
  net.add_layer(4, "linear");
  net.add_layer(5, "sigmoid");
  net.set_activation(1, lecun_tanh, lecun_tanh_deriv);
  net.add_layer(1, "resig");
  net.initialize();
  net.list_net();
  net.train(50);
  //  net.list_net();
}
