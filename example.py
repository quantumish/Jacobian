import mrbpnn
import matplotlib.pyplot as plt
import numpy
import time

batch_sz = 10
layers = 2
epochs = 50
lr = 0.0155
bias_lr = 0.03
neurons = 3

net = mrbpnn.Network("./data_banknote_authentication.txt", batch_sz, lr, bias_lr)
net.add_layer(4, "linear")
for i in range(layers):
    net.add_layer(neurons, "lecun_tanh")
net.add_layer(1, "resig")
net.initialize()

import wandb
wandb.init(project="jacobian")
wandb.config.update({"epochs": epochs,
                     "batch_size": batch_sz,
                     "learning_rate": lr,
                     "bias_lr": bias_lr,
                     "hidden_layers": layers,
                     "activation":"lecun_tanh",
                     "neurons": neurons})
for i in range(epochs):
    net.train(1)
    wandb.log({'accuracy': net.get_acc(), 'cost': net.get_cost()})

wandb.save('jacobian.h5')
