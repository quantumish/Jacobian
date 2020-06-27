import mrbpnn
import numpy
import time

init = time.time()
net = mrbpnn.Network("./exoplanets.txt", 10, 0.0155, 0.03);
net.add_layer(4, "sigmoid");
net.add_layer(5, "step");
net.add_layer(5, "step");
net.add_layer(5, "lecun_tanh");
net.add_layer(1, "resig");
net.initialize();
initend = time.time()
net.train(50);

end = time.time()
print("%s: init %s" % (end-init, initend-init))

