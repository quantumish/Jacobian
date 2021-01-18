from sklearn.neural_network import MLPClassifier
import csv
import matplotlib.pyplot as plt
import time
import numpy as np

def bench(batch_sz, layers, layer_size, epochs, lr):
    start = time.time()
    with open("./data_banknote_authentication.txt", 'rt') as f:
        reader = csv.reader(f)
        data = list(reader)
        for a in data:
            for b, c in enumerate(a):
                a[b] = float(a[b])
            
    X_train = []
    y_train = []
    for i in data:
        X_train.append(i[:-1])
        y_train.append(i[-1])

    clf = MLPClassifier(solver="sgd", momentum=0, learning_rate_init=lr, batch_size=batch_sz, max_iter=1, hidden_layer_sizes=(layer_size), warm_start=True)
    accuracy = []1p
    for i in range(50):
        clf.fit(X_train, y_train)
        #accuracy.append(clf.score(X_train,y_train))
 
    end = time.time()
    #print(clf.loss_curve_)
    #print(clf.validation_scores_)
    #print(accuracy)
    #print(len(clf.loss_curve_), len(accuracy))
    return end-start

print(bench(16))    
