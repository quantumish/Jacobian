#
#  kerasdemo.py
#  Jacobian
#
#  Created by David Freifeld
#  Copyright © 2020 David Freifeld. All rights reserved.
#

#+-----------------------------------------------------------------------------+
# Keras benchmark code to compare with network.
#
# Runs a fully connected feedforward neural network with backpropagation for 50
# epochs, then tests on data. First layer has 4 neurons, 2 hidden layers have 5,
# output layer has 1. All four layers use sigmoid for activation.
#
# Taken and loosely modified from:
# https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
#+-----------------------------------------------------------------------------+

import time
import numpy
# import tensorflow
from numpy import loadtxt
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers import Dense

def kerasbench(batch_sz, layers):
    init = time.time()
    dataset = loadtxt('data_banknote_authentication.txt', delimiter=',')
    X = dataset[:,0:4]
    y = dataset[:,4]
    model = Sequential()
    model.add(Dense(4, input_dim=4, activation='linear'))
    for i in range(layers):
        model.add(Dense(5, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    opt = keras.optimizers.SGD(lr=0.0155)
    model.compile(loss='mse', optimizer=opt, metrics=['accuracy'])
    model.fit(X, y, epochs=50, batch_size=batch_sz)
    end = time.time()
    return (end-init)

# sum = 0
# for i in range(10):
#     sum += kerasbench(10, 1)
# print(sum/10)
    
y2 = []
y2.append(kerasbench(1, 1))
y2.append(kerasbench(5, 1))
y2.append(kerasbench(10, 1))
y2.append(kerasbench(15, 1))
print(y2)
