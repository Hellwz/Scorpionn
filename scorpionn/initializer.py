import numpy as np

def init_zeros(shape):
    return np.full(shape=shape, fill_value=0).astype(np.float32)

def init_ones(shape):
    return np.full(shape=shape, fill_value=1).astype(np.float32)