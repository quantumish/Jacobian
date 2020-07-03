import mrbpnn
import numpy as np
import matplotlib.pyplot as plt
import numpy
import time
import wandb
from viznet import NodeBrush, EdgeBrush, DynamicShow

def draw_feed_forward(ax, num_node_list):
    '''
    draw a feed forward neural network.

    Args:
        num_node_list (list<int>): number of nodes in each layer.
    '''
    num_hidden_layer = len(num_node_list) - 2
    token_list = ['\sigma^z'] + \
        ['y^{(%s)}' % (i + 1) for i in range(num_hidden_layer)] + ['\psi']
    kind_list = ['nn.input'] + ['nn.hidden'] * num_hidden_layer + ['nn.output']
    radius_list = [0.3] + [0.2] * num_hidden_layer + [0.3]
    y_list = 1.5 * np.arange(len(num_node_list))

    seq_list = []
    for n, kind, radius, y in zip(num_node_list, kind_list, radius_list, y_list):
        b = NodeBrush(kind, ax)
        seq_list.append(node_sequence(b, n, center=(0, y)))

    eb = EdgeBrush('-->', ax)
    for st, et in zip(seq_list[:-1], seq_list[1:]):
        connecta2a(st, et, eb)


def real_bp():
    with DynamicShow((6, 6), '_feed_forward.png') as d:
        draw_feed_forward(d.ax, num_node_list=list)


batch_sz = 10
layers = 1
epochs = 50
lr = 0.0155
bias_lr = 0.03
neurons = 10
ratio = 0.9
layers_list = []

init = time.time()
net = mrbpnn.Network("./data_banknote_authentication.txt", batch_sz, lr, bias_lr, ratio)
net.add_layer(4, "linear")
layers_list.append(4)
for i in range(layers):
    net.add_layer(neurons, "lecun_tanh")
    layers_list.append(neurons)
net.add_layer(1, "resig")
layers_list.append(1)
net.initialize()

wandb.init(project="jacobian")
wandb.config.update({"epochs": epochs,
                     "batch_size": batch_sz,
                     "learning_rate": lr,
                     "bias_lr": bias_lr,
                     "hidden_layers": layers,
                     "activation":"lecun_tanh",
                     "neurons": neurons,
                     "data_split":ratio})

for i in range(epochs):
    net.train(1)
    wandb.log({'accuracy': net.get_acc(), 'cost': net.get_cost(), 'val_accuracy': net.get_val_acc(), 'val_cost': net.get_val_cost()})
end = time.time()
    
wandb.run.summary["time"] = end-init
wandb.save('jacobian.h5')

with DynamicShow((6, 6), '_feed_forward.png') as d:
        draw_feed_forward(d.ax, num_node_list=layers_list)
