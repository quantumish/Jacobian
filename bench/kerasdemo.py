#
#  kerasdemo.py
#  Jacobian
#
#  Created by David Freifeld
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
import matplotlib.pyplot as plt
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
    model.add(Dense(1, activation='linear'))
    opt = keras.optimizers.SGD(lr=0.0155)
    model.compile(loss='mse', optimizer=opt, metrics=['accuracy'])
    history = model.fit(X, y, epochs=50, validation_split=0.1, batch_size=batch_sz)
    end = time.time()
    print(history.history['loss'])
    print(history.history['acc'])
    print(history.history['val_loss'])
    print(history.history['val_acc'])
    return (end-init)

# sum = 0
# for i in range(10):
#     sum += kerasbench(10, 1)
# print(sum/10)
    
print(kerasbench(16,1))
