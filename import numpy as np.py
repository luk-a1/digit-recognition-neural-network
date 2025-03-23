import random
import numpy as np

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

class Network(object):
    
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.weights = [np.random.randn(y,x) for x,y in zip(sizes[:-1], sizes[1:])]
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]

    def feedforward(self, a):
        for w,b in zip(self.weights, self.biases):
            a = sigmoid(np.dot(w, a) + b)