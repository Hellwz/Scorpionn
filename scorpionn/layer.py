import numpy as np
from scorpionn import initializer

class Base_Layer:
    def __init__(self, type):
        self.type = type # type of the layer
        self.params = None
        self.grads = None

    def forward(self, inputs_p): # inputs_propogate
        raise NotImplementedError

    def backward(self, grads_p): # grad_propogate
        raise NotImplementedError

class Base_Activation(Base_Layer):
    def __init__(self, type):
        super().__init__(type)
        self.inputs = None

    def forward(self, inputs_p):
        self.inputs = inputs_p
        return self.act_func(inputs_p)

    def backward(self, grads_p):
        return grads_p * self.der_func(self.inputs) 

    def act_func(self, x): # activation function
        raise NotImplementedError

    def der_func(self, x): # derivative function
        raise NotImplementedError

class FC(Base_Layer):
    def __init__(self, num_in, num_out, 
                 weight_init=initializer.init_xavier_uniform,
                 bias_init=initializer.init_zeros):
        super().__init__('FC')
        self.inputs = None
        self.params = {
            "weight": weight_init([num_in, num_out]),
            "bias": bias_init([1, num_out])
        }
        self.grads = {}
    
    def forward(self, inputs_p):
        self.inputs = inputs_p # shape: (batch_size, num_in)
        # z[i] = a[i-1] * w[i] + b[i]
        return inputs_p @ self.params["weight"] + self.params["bias"]

    def backward(self, grads_p):
        # dw[i] = a[i-1].T * dz[i]
        self.grads["weight"] = self.inputs.T @ grads_p # partial derivative to w
        self.grads["bias"] = np.sum(grads_p, axis=0) # partial derivative to b
        return grads_p @ self.params["weight"].T # partial derivative to inputs

class ReLU(Base_Activation):
    def __init__(self):
        super().__init__('ReLU')

    def act_func(self, x):
        return np.maximum(x, 0.0)

    def der_func(self, x):
        return x > 0.0

class Sigmoid(Base_Activation):
    def __init__(self):
        super().__init__('Sigmoid')

    def act_func(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def der_func(self, x):
        return self.act_func(x) * (1.0 - self.act_func(x))