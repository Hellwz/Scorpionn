import numpy as np

def init_zeros(shape):
    return np.full(shape=shape, fill_value=0).astype(np.float32)

def init_ones(shape):
    return np.full(shape=shape, fill_value=1).astype(np.float32)
    # Not suitable for initialization of FC layer's weights
    # Cause grads problems

def init_xavier_uniform(shape):
    a = np.sqrt(6. / (shape[0] + shape[1]))
    return np.random.uniform(low=-a, high=a, size=shape).astype(np.float32)
    