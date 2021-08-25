import numpy as np

class Base_Loss:
    def __init__(self, name):
        self.name = name

    def get_loss(self, y_hat, y):
        raise NotImplementedError

    def get_grads(self, y_hat, y):
        raise NotImplementedError

class SoftmaxCrossEntropyLoss(Base_Loss):
    def __init__(self):
        super().__init__('CrossEntropyLoss')

    def get_loss(self, y_hat, y):
        batch_size = y_hat.shape[0]
        return -np.sum(y * np.log(self.softmax(y_hat))) / batch_size
        # or np.mean(-np.sum(y * np.log(self.softmax(y_hat)), axis=1))

    def get_grads(self, y_hat, y):
        batch_size = y_hat.shape[0]
        return (self.softmax(y_hat) - y) / batch_size
        # average loss and grads are what we need

    def softmax(self, X):
        maxx = np.max(X) # prevent overflow
        X_exp = np.exp(X - maxx)
        return X_exp / np.sum(X_exp, axis=1, keepdims=True)


    