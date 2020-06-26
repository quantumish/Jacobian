import mrbpnn
import numpy
import time

def lecun_tanh(x):
  return 1.7159 * numpy.tanh((2.0/3) * x)

def lecun_tanh_deriv(x):
  return 1.14393 * (1.0/numpy.cosh(2.0/3 * x))**2

init = time.time()
net = mrbpnn.Network("./data_banknote_authentication.txt", 10, 0.01, 0.001);
net.add_layer(4, "linear");
net.add_layer(5, "sigmoid");
net.set_activation(1, lecun_tanh, lecun_tanh_deriv);
net.add_layer(1, "sigmoid");
net.initialize();
net.train(50);
end = time.time()
print(end-init)

