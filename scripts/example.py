import mrbpnn
import numpy as np
import matplotlib.pyplot as plt
import numpy
import time
import wandb

data_split = 0.75

hyperparameter_defaults = dict(batch_size = 10,
                               hidden_layers = 1,
                               epochs = 50,
                               learning_rate = 0.0155,
                               bias_lr = 0.03,
                               activation = "lecun_tanh",
                               neurons = 10,
                               l = 0)

wandb.init(project="jacobian", config=hyperparameter_defaults)
config = wandb.config

init = time.time()
net = mrbpnn.Network("../data_banknote_authentication.txt", config.batch_size, config.learning_rate, config.bias_lr, config.l, data_split)
net.add_layer(4, "linear")
for i in range(config.hidden_layers):
    net.add_layer(config.neurons, config.activation)
net.add_layer(1, "resig")
net.initialize()

for i in range(config.epochs):
    net.train()
    wandb.log({'accuracy': net.get_acc(), 'cost': net.get_cost(), 'val_accuracy': net.get_val_acc(), 'val_cost': net.get_val_cost()})
end = time.time()
    
wandb.run.summary["time"] = end-init
wandb.save('jacobian.h5')
