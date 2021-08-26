import os
import struct
import numpy as np

def load_mnist(path='..\\datasets\\mnist', type='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path, '%s-labels.idx1-ubyte' % type)
    images_path = os.path.join(path, '%s-images.idx3-ubyte' % type)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)

    num_class = 10
    labels_onehot = np.zeros((len(labels), num_class))
    labels_onehot[range(len(labels)), labels] = 1

    return images, labels_onehot