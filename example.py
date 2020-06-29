import mrbpnn
import numpy
import time

def bench():
    init = time.time()
    net = mrbpnn.Network("./data_banknote_authentication.txt", 10, 0.001, 0.03);
    net.add_layer(4, "linear");
    net.add_layer(5, "relu");
    net.add_layer(1, "resig");
    net.initialize();
    initend = time.time()
    net.train(50);
    end = time.time()
    return (end-init)
#    print("%s: init %s" % (end-init, initend-init))

# timesum=0
# trials = 1000
# for i in range(trials):
#     timesum+=bench()
# print("Averages over %s trials\n--------------\nTime: %s seconds.\n" % (trials, timesum/trials))
print(bench())
