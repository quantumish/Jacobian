import jacobian as jcb
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import time
net = jcb.Network("./data_banknote_authentication.txt", 10, 0.0155, 0.03, jcb.L2, 1, 0.9)
net.add_layer(4, jcb.activations.linear, jcb.activations.linear_deriv)
net.add_layer(5, jcb.activations.sigmoid, jcb.activations.sigmoid_deriv)
net.add_layer(2, jcb.activations.linear, jcb.activations.linear_deriv)
net.init_optimizer(jcb.optimizers.momentum(0.1))
net.init_decay(jcb.decays.exponential(1, 0.5))
net.initialize()
for i in range(50):
    net.train()
