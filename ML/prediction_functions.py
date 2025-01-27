import numpy as np

def linear(X, w, b, all_examples=False):
    if all_examples == True:
        return np.matmul(X, w) + b
    else:
        return np.dot(X, w) + b #here X would be a single example
    
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def softmax(z):
    softmaxed = np.exp(z)
    softmaxed = softmaxed/np.sum(softmaxed)
    return softmaxed