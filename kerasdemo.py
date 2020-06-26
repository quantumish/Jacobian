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
def lecun_tanh(x):
    return 1.7159 * K.tanh((2.0/3) * x)

init = time.time()
# load the dataset
dataset = loadtxt('extra.txt', delimiter=',')
# split into input (X) and output (y) variables
X = dataset[:,0:4]
y = dataset[:,4]
# define the keras model
model = Sequential()
model.add(Dense(4, input_dim=4, activation='linear'))
model.add(Dense(5, activation=lecun_tanh))
model.add(Dense(1, activation='sigmoid'))
# compile the keras model
opt = keras.optimizers.SGD(lr=0.01)
model.compile(loss='mse', optimizer=opt, metrics=['accuracy'])
# fit the keras model on the dataset
initend = time.time()
model.fit(X, y, epochs=50, batch_size=10)
# evaluate the keras model
#_, accuracy = model.evaluate(X, y)
#print('Accuracy: %.2f' % (accuracy*100))
end = time.time()
print("%s: init %s" % (end-init, initend-init))
