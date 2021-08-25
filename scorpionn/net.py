class MyNet:
    def __init__(self, name, layers):
        self.name = name
        self.layers = layers

    def forward(self, inputs_p):
        for layer in self.layers:
            inputs_p = layer.forward(inputs_p)
        return inputs_p

    def backward(self, grads_p):
        grads_rec = [] # record the grads
        for layer in reversed(self.layers):
            grads_p = layer.backward(grads_p)
            grads_rec.append(grads_p)
        return grads_rec[::-1] # reverse the records

    def get_params_and_grads(self):
        for layer in self.layers:
            yield layer.params, layer.grads

