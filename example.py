import mrbpnn
import numpy
import time

init = time.time()
net = mrbpnn.Network("./data_banknote_authentication.txt", 10, 0.0155, 0.0155);
net.add_layer(4, "linear");
net.add_layer(5, "sigmoid");
net.set_activation(1, lecun_tanh, lecun_tanh_deriv);
net.add_layer(1, "sigmoid");
net.initialize();
initend = time.time()
net.train(50);
end = time.time()
print("%s: init %s" % (end-init, initend-init))

