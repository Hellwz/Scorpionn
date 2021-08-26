import os,sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,parentdir)

from scorpionn import layer, loss, net, model, optimizer, initializer
import numpy as np

# layer1 = layer.FC(10, 20)

# in1 = np.full(shape=[1,10], fill_value=1).astype(np.float32)
# # print(in1.dtype)
# print(layer1.forward(in1))
# print(in1)

# layer2 = layer.Sigmoid()
# print(layer2.act_func(in1).dtype)

# np1 = np.array([[10, 1, 1], [1, 10, 1], [1, 1, 10]])
# np2 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
# loss1 = loss.SoftmaxCrossEntropyLoss()
# print(loss1.get_loss(np1, np2))

'''
# Test the difference between
# 'params += grads' and 'params = params + grads' in function.
# The latter will create a new 'params'.
# But the case will be different in a dictionary.

in1 = np.full(shape=[1,5], fill_value=1).astype(np.float32)
in2 = {'1': np.full(shape=[1,5], fill_value=1).astype(np.float32)}

def test(in1, in2):
    in1 *= 2 
    # will change 'in1' outside
    in1 = 2 * in1 
    # won't change 'in1' outside
    for key in in2.keys():
        in2[key] *= 2 
        # will change 'in2' outside
        in2[key] = 2 * in2[key] 
        # will change 'in2' outside

test(in1, in2)
print(in1, in2)
'''

# layer1 = layer.FC(10, 20, weight_init=initializer.init_ones,
#                  bias_init=initializer.init_ones)
# layer2 = layer.FC(20, 5, weight_init=initializer.init_ones,
#                  bias_init=initializer.init_ones)
# net1 = net.MyNet('net1', [layer1, layer2])
# in1 = np.full(shape=[1,10], fill_value=1).astype(np.float32)
# y1 = np.array([1, 0, 0, 0, 0])
# model1 = model.MyModel('model1', net1, optimizer.SGD(), loss.SoftmaxCrossEntropyLoss())
# out1 = model1.forward(in1)
# loss, grads_rec = model1.backward(out1, y1)
# print(out1)
# print(loss, grads_rec)

print(initializer.init_xavier_uniform([10,20]))