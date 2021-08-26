class Base_Optimizer():
    def __init__(self, name, lr):
        self.name = name
        self.lr = lr

    def apply_step(self, params, grads):
        if params == None and grads == None:
            return
        grads = self.compute_step(grads)
        for key in params.keys():
            params[key] += grads[key]

    def compute_step(self, grads):
        raise NotImplementedError

class SGD(Base_Optimizer):
    def __init__(self, name='SGD', lr=0.001):
        super().__init__(name, lr)
    
    def compute_step(self, grads):
        for key in grads.keys():
            grads[key] = -self.lr * grads[key]
        return grads
