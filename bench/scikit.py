from sklearn.neural_network import MLPClassifier
import csv
import matplotlib.pyplot as plt
import time

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

    clf = MLPClassifier(solver="sgd", batch_size=batch_sz, hidden_layer_sizes=(5))
    clf.fit(X_train, y_train)
    end = time.time()
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
