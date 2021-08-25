class MyModel:
    def __init__(self, name, net, optimizer, loss):
        self.name = name
        self.net = net
        self.optimizer = optimizer
        self.loss = loss
    
    def forward(self, inputs):
        return self.net.forward(inputs)

    def backward(self, y_hat, y):
        loss = self.loss.get_loss(y_hat, y)
        grads_l = self.loss.get_grads(y_hat, y) # grads from loss
        grads_rec = self.net.backward(grads_l)
        return loss, grads_rec
    
    def apply_grads(self):
        self.optimizer.apply_step(self.net.get_params_and_grads())