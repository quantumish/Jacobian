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
# import tensorflow
from numpy import loadtxt
import keras
from keras.models import Sequential
from keras.layers import Dense
init = time.time()
# load the dataset
dataset = loadtxt('data_banknote_authentication.txt', delimiter=',')
# split into input (X) and output (y) variables
X = dataset[:,0:4]
y = dataset[:,4]
# define the keras model
model = Sequential()
model.add(Dense(4, input_dim=4, activation='sigmoid'))
model.add(Dense(5, activation='sigmoid'))
model.add(Dense(5, activation='sigmoid'))
model.add(Dense(1, activation='relu'))
# compile the keras model
opt = keras.optimizers.SGD(lr=1)
model.compile(loss='mse', optimizer=opt, metrics=['accuracy'])
# fit the keras model on the dataset
model.fit(X, y, epochs=50, batch_size=1)
# evaluate the keras model
_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))
end = time.time()
print(end-init)
