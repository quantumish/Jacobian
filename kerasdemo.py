# first neural network with keras tutorial
import time
# import tensorflow
from numpy import loadtxt
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
model.add(Dense(1, activation='sigmoid'))
# compile the keras model
model.compile(loss='mse', optimizer='sgd', metrics=['accuracy'])
# fit the keras model on the dataset
model.fit(X, y, epochs=50, batch_size=10)
# evaluate the keras model
_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))
end = time.time();
print(end-init);
