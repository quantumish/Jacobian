from sklearn.neural_network import MLPClassifier
import csv

with open("./data_banknote_authentication.txt", 'rt') as f:
    reader = csv.reader(f)
    data = list(reader)
    for a in data:
        for b, c in enumerate(a):
            a[b] = float(a[b])

X_train = []
y_train = []
for i in data:
    X_train.append(data[:-1])
    y_train.append(data[-1])
print(X_train[-1])

clf = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)
