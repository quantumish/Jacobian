from sklearn.neural_network import MLPClassifier
import csv
import matplotlib.pyplot as plt
import time
import numpy as np

def bench(batch_sz):
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

    clf = MLPClassifier(solver="sgd", momentum=0, learning_rate_init=0.0155, batch_size=batch_sz, max_iter=1, hidden_layer_sizes=(5), warm_start=True)
    accuracy = []
    for i in range(50):
        clf.fit(X_train, y_train)
        accuracy.append(clf.score(X_train,y_train))
 
    end = time.time()
    print(clf.loss_curve_)
    #print(clf.validation_scores_)
    print(accuracy)
    print(len(clf.loss_curve_), len(accuracy))
    return end-start

# i = 1
# while (i < 1343):
#     times = []
#     times.append(bench(i))
#     print("Finished loop %s in %s s." %(i, times[-1]))
#     if (i == 1): i += 9
#     else: i += 10
# print(times)
print(bench(16))    
