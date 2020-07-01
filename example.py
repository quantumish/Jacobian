import mrbpnn
import matplotlib.pyplot as plt
import numpy
import time

batch_sz = 10
layers = 1

net = mrbpnn.Network("./data_banknote_authentication.txt", batch_sz, 0.0155, 0.03)
net.add_layer(4, "linear")
for i in range(layers):
    net.add_layer(5, "relu")
net.add_layer(1, "resig")
net.initialize()

import wandb
wandb.init(project="jacobian")
wandb.config.update({"epochs": 50, "batch_size": batch_sz, "hidden_layers": layers})
for i in range(50):
    net.train(1)
    wandb.log({'accuracy': net.get_info()})

wandb.save('jacobian.h5')
