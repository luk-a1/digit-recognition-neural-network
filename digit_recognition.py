"""http://neuralnetworksanddeeplearning.com/chap1.html"""
import random
import numpy as np

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

class Network(object):
    
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.weights = [np.random.randn(y,x) for x,y in zip(sizes[:-1], sizes[1:])]
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]

    def feedforward(self, a):
        for w,b in zip(self.weights, self.biases):
            a = sigmoid(np.dot(w, a) + b)
        
        return a

    def sgd(self, training_data, epochs, mini_batch_size, eta, test_data = None):

        if test_data:
            test = len(test_data)

        n = len(training_data)

        for i in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            
            for mini_batch in mini_batches:
                self.update(mini_batch, eta)
            
            if test_data:
                print("Epoch {0}: {1} / {2}".format(i, self.evaluate(test_data), test))
            else:
                print ("Epoch {0} complete".format(i))
    
    def update(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.biases]

        for x, y in mini_batch:
            delta_w, delta_b = self.back_propagation(x,y)
            nabla_b = [b + db for b, db in zip(nabla_b, delta_b)]
            nabla_w = [w + dw for w, dw in zip(nabla_w, delta_w)]
        
        self.weights = [w - eta/len(mini_batch) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - eta/len(mini_batch) * nb for b, nb in zip(self.biases, nabla_b)]
    
    def back_propagation(self, x, y):
        delta_b = [np.zeros(b.shape) for b in self.biases]
        delta_w = [np.zeros(w.shape) for w in self.weights]

        activation = x
        activations = [x]
        zs = []

        #feedforward
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        delta = (activations[-1] - y) * sigmoid_prime(zs[-1])
        delta_b[-1] = delta #equation 3
        delta_w[-1] = np.dot(delta, activations[-2].transpose()) #equation 4

        for l in (2, self.num_layers):
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sigmoid_prime(zs[-l])
            delta_b[-l] = delta #equation 3
            delta_w[-l] = np.dot(delta, activations[-l-1].transpose()) #equation 4
        
        return (delta_b, delta_w)

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)





