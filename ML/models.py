import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ML.gradients
from ML.costs import MeanSquaredError, BinaryCrossEntropy
from utils.utils import timer
from collections import Counter


'''
All of the functions assume that 2d matrices get passed . e.g (2, 1) instead of (2, ))
'''
class LinearRegressor:

    def __init__(self, learning_rate=0.00001, num_iters=100, regul=False, regul_type="l2", lambda_=1):
        self.learning_rate = learning_rate
        self.num_iters = num_iters
        self.regul = regul
        self.regul_type = regul_type
        self.lambda_ = lambda_
        self.weights = 0 
        self.bias = 0

    def gradient(self, X, y):
        '''
        X --> (m x n)
        y --> (m x 1)
        W --> (n x 1)
        b --> int/float...
        '''

        no_of_examples, no_of_features = X.shape 


        dj_dw = np.zeros((no_of_features, 1)) #gradient of cost function wrt w (weights)
        dj_db = 0 #gradient of cost function wrt b (bias)

        err = (np.matmul(X, self.weights) + self.bias) - y 

        dj_dw_tmp = np.sum(X*err, axis=0) 
        dj_dw = dj_dw_tmp.T

        dj_db = np.sum(err)

        dj_dw = dj_dw/no_of_examples
        dj_db = dj_db/no_of_examples

        if self.regul == True:
            if self.regul_type == "l2":
                dj_dw = dj_dw + ((self.lambda_/no_of_examples) * self.weights)

        return dj_dw, dj_db
    

    #gradient descent
    def fit(self, X, y, return_cost_history=False, verbose=False):

        no_of_features = X.shape[1]

        self.weights = np.zeros((no_of_features, 1))

        if return_cost_history == True:
            cost_history = np.zeros((self.num_iters, 1))

        for i in range(self.num_iters):

            dj_dw, dj_db = self.gradient(X, y)

            if np.linalg.norm(dj_dw) < 1e-6 and abs(dj_db) < 1e-6:
                break

            #increment w and b
            self.weights = self.weights - self.learning_rate*dj_dw
            self.bias = self.bias - self.learning_rate*dj_db

            if return_cost_history == True:
                cost_history[i][0] = MeanSquaredError(X, y, self.weights, self.bias, regul=self.regul, regul_type=self.regul_type, lambda_=self.lambda_)

            if verbose == True and i % 100 == 0:
                print(f"W: {self.weights}; b: {self.bias}; cost: {self.cost(X, y)}; num_iters: {i}")

        if return_cost_history == True:
            return self.weights, self.bias, cost_history
        
        if self.weights == np.nan or self.bias == np.nan:
            print("Couldn't converge")

        return self.weights, self.bias  
    


    def predict(self, X):
        return (np.matmul(X, self.weights) + self.bias)






class LogisticRegressor:

    def __init__(self, learning_rate=0.001, num_iters=100, regul=False, regul_type="l2", lambda_=1):
        self.learning_rate = learning_rate
        self.num_iters = num_iters
        self.regul = regul
        self.regul_type = regul_type
        self.lambda_ = lambda_
        self.weights = 0 
        self.bias = 0

    def gradient(self, X, y):
        '''
        X --> (m x n)
        y --> (m x 1)
        W --> (n x 1)
        b --> int/float...
        '''

        no_of_examples, no_of_features = X.shape 


        dj_dw = np.zeros((no_of_features, 1)) #gradient of cost function wrt w (weights)
        dj_db = 0 #gradient of cost function wrt b (bias)

        err = ML.prediction_functions.sigmoid(np.matmul(X, self.weights) + self.bias) - y 

        dj_dw_tmp = np.sum(X*err, axis=0) 
        dj_dw = dj_dw_tmp.T

        dj_db = np.sum(err)

        dj_dw = dj_dw/no_of_examples
        dj_db = dj_db/no_of_examples

        if self.regul == True:
            if self.regul_type == "l2":
                dj_dw = dj_dw + ((self.lambda_/no_of_examples) * self.weights)

        return dj_dw, dj_db


    def fit(self, X, y, return_cost_history=False, verbose=False):
        '''
        X --> (m x n)
        y --> (m x 1)
        W --> (n x 1)
        b --> int/float...
        '''
        self.weights = np.zeros((X.shape[1], 1))

        if return_cost_history == True:
            cost_history = np.zeros((self.num_iters, 1))

        for i in range(self.num_iters):

            dj_dw, dj_db = self.gradient(X, y)

            if np.linalg.norm(dj_dw) < 1e-6 and abs(dj_db) < 1e-6:
                break

            #increment w and b
            self.weights = self.weights - self.learning_rate*dj_dw
            self.bias = self.bias - self.learning_rate*dj_db

            if return_cost_history == True:
                cost_history[i][0] = BinaryCrossEntropy(X, y, self.weights, self.bias, regul=self.regul, regul_type=self.regul_type, lambda_=self.lambda_)

            if verbose == True and i % 100 == 0:
                print(f"W: {self.weights}; b: {self.bias}; cost: {self.cost(X, y)}; num_iters: {i}")

        if return_cost_history == True:
            return self.weights, self.bias, cost_history
        
        if self.weights == np.nan or self.bias == np.nan:
            print("Couldn't converge")

        return self.weights, self.bias
    


    def predict(self, X):
        return (np.matmul(X, self.weights) + self.bias)



class KNN:

    def __init__(self, no_of_neighbors=10):
        self.X = np.nan
        self.y = np.nan
        self.no_of_neighbors = no_of_neighbors

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X):
        prediction = np.empty(X.shape[0], dtype=str)
        for i in range(X.shape[0]):
            distance = np.sqrt(np.sum((np.square(self.X - X[i])), axis=1))
            distance = (np.vstack((distance, self.y))).T
            distance = (distance[distance[:, 0].argsort()])[:self.no_of_neighbors]
            voting = Counter(distance[:, 1])
            prediction[i] = (voting.most_common(1))[0][0]

        return prediction
    
    


class SoftmaxRegressor:

    def __init__(self):
        pass

    def fit(self, X , y):
        pass

    def predict(self, X):
        pass