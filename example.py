import jacobian as jcb
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import time
net = jcb.Network("./data_banknote_authentication.txt", 10, 0.0155, 0.03, jcb.L2, 1, 0.9)
net.add_layer(4, jcb.linear, jcb.linear_deriv)
net.add_layer(5, jcb.sigmoid, jcb.sigmoid_deriv)
net.add_layer(2, jcb.linear, jcb.linear_deriv)
net.init_optimizer(jcb.momentum(0.1))
net.init_decay(jcb.exponential(1, 0.5))
net.initialize()
fig, axs = plt.subplots()
accuracies = []
costs = []
def update(frame):
    net.feedforward()
    net.backpropagate()
    for i in range(3):
        plt.subplot(3, 4, 4*i + 1)
        plt.imshow(net.layers[i].get_contents())
        plt.colorbar()
        plt.subplot(3, 4, 4*i + 2)
        plt.imshow(net.layers[i].get_weights())
        plt.colorbar()
        plt.subplot(3, 4, 4*i + 3)
        plt.imshow(net.layers[i].get_bias())
        plt.colorbar()
    accuracies.append(net.accuracy())
    costs.append(net.cost())
    plt.subplot(3,4,4)
    plt.plot(costs[1:], color="blue")
    plt.subplot(3,4,8)
    plt.plot(accuracies[1:], color="orange")
    time.sleep(0.01)
    net.next_batch()
ani = FuncAnimation(fig, update, interval=1)
plt.show()
