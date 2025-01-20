import numpy as np
import functions

def MeanSquaredError(X: np.ndarray, y: np.ndarray, W: np.ndarray, b: np.float64, prediction_function=functions.linear, regul: bool=False, lambda_: np.float64=1.0) -> np.float64:
    no_of_examples: int = X.shape[0]

    total_cost: np.float64 = 0
    reg_cost:   np.float64 = 0 #regularization cost

    total_cost = np.square((prediction_function(X, W, b)) - y)

    if regul == True:
        reg_cost   = (np.sum(W, axis=0))**2
        reg_cost   = (lambda_/(2*no_of_examples)) * reg_cost

    return total_cost + reg_cost   

def BinaryCrossEntropy(X: np.ndarray, y:np.ndarray, W: np.ndarray, b: np.ndarray, function=functions.sigmoid, regul=False, lambda_=1) -> np.float64:
    no_of_examples: int = X.shape[0]

    total_cost: np.float64 = 0
    reg_cost:   np.float64 = 0 #regularization cost

    sigmod_value: np.ndarray = function(functions.linear(X, W, b))
    total_cost = -y*np.log(sigmod_value) - (1-y)*np.log(1-sigmod_value)
    total_cost = np.sum(total_cost, axis=0)

    if regul == True:
        reg_cost   = (np.sum(W, axis=0))**2
        reg_cost   = (lambda_/(2*no_of_examples)) * reg_cost

    return total_cost + reg_cost
    



