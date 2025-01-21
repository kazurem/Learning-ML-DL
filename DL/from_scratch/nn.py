import numpy as np
import matplotlib.pyplot as plt
from icecream import ic

'''
In the making
'''




'''
input data --> (m x n)
'''


class Neuron:
    
    def __init__(self, activation, weights=0, bias=0):
        self.weights = weights
        self.bias = bias
        self.activation = activation

    def compute(self, input):
        '''
        weights --> (1 x n)
        input vector --> (1 x n)
        w . a --> 1
        '''
        output = self.activation(np.dot(self.weights, input) + self.bias)
        return output


class Layer:
    
    def __init__(self, units, activation, name, weights=0, biases=0):
        self.units = units
        self.activation = activation
        self.weights = weights
        self.biases = biases
        self.name = name

    def compute(self, input):
        ic(self.weights.shape, input.shape)
        a = np.zeros((self.units, 1))
        if self.activation == "linear":
            if self.weights.shape[0] == 1 and self.weights.shape[1] == 1:
                self.weights = self.weights.reshape(-1, )
            for i in range(self.units):
                output = np.dot(input, self.weights[i])
                ic(output)
                a[i] = np.dot(input, self.weights[i]) + self.biases[i]
        # if self.activation == "linear":
        #     if self.weights.shape[0] == 1 and self.weights.shape[1] == 1:
        #         output = np.dot(self.weights, input) + self.biases
        #     else:
        #         output = (np.dot(self.weights, input) + self.biases)
            
        return a


class mySequential:
    
    def __init__(self, layers, weights=0, biases=0):
        self.layers = layers
        self.weights = weights
        self.biases = biases

        # self.weightsAndBiasesIntoLayers()


    #Initializing the model's weights and biases
    def weightsAndBiasesIntoLayers(self):
        for layer in self.layers:
            layer.weights = self.weights[layer]
            layer.biases = self.biases[layer]

    def setWeightsAndBiases(self, weights, biases):
        for layer in self.layers:
            layer.weights = weights[layer]
            layer.biases  = biases[layer]

    def compile(self, loss, optimizer, metrics):
        pass

    #backward propagation
    def fit(self, X_train, y_train):
        pass

    #forward propagation
    def predict(self, input):
        output = self.layers[0].compute(input)
        for i, layer in enumerate(self.layers):
            if i == 0:
                continue
            output = layer.compute(output)
        return output





weights_l1 = np.array([
    [2.3, 4.1, 1.2],
])
weights_l2 = np.array([
    [1],
])
biases_l1 = np.array([
    [1, 2, 3],
])
biases_l2 = np.array([
    [1],
])

input = np.array([
    [1],
])

model = mySequential(
    [
        Layer(units=3, activation="linear", name="l1", weights=weights_l1[0].reshape(-1, 1), biases=biases_l1[0].reshape(-1, 1)),
        Layer(units=1, activation="linear", name="l2", weights=weights_l2[0].reshape(-1, 1), biases=biases_l2[0].reshape(-1, 1)),
    ],
)

output = model.predict(input)
print(output)



    