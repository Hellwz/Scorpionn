import os,sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,parentdir)

import scorpionn as snn
import numpy as np
import math

layers = []
layers.append(snn.layer.FC(784, 500))
layers.append(snn.layer.ReLU())
layers.append(snn.layer.FC(500, 100))
layers.append(snn.layer.ReLU())
layers.append(snn.layer.FC(100, 10))

net1 = snn.net.MyNet('net1', layers)
model1 = snn.model.MyModel('model1', net1, snn.optimizer.SGD(lr=0.01), snn.loss.SoftmaxCrossEntropyLoss())
epoch = 10
batch_size = 32

X_train, Y_train = snn.dataset.load_mnist(type='train')
X_test, Y_test = snn.dataset.load_mnist(type='t10k')
X_train = X_train * 1.0 / 255
X_test = X_test * 1.0 / 255

for i in range(epoch):
    print("epoch:", i, "is running")
    batch_num = math.ceil(X_train.shape[0] * 1.0 / batch_size)
    train_loss = 0
    train_acc = 0
    for i in range(batch_num):
        l = i * batch_size
        r = min((i+1) * batch_size, X_train.shape[0])
        X = X_train[l:r]
        Y = Y_train[l:r]
        outputs = model1.forward(X)
        # print(outputs)
        loss, grads_rec = model1.backward(outputs, Y)
        model1.apply_grads()

        pred_class = np.argmax(outputs, axis=1)
        true_class = np.argmax(Y, axis=1)
        accuracy, info = snn.accuracy.accuracy_classification(pred_class, true_class)
        train_loss += loss
        train_acc += accuracy
        # print(loss, accuracy, info)
    
    train_loss /= batch_num
    train_acc /= batch_num
    print("loss:", train_loss, "acc:", train_acc)


