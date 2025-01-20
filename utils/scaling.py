import numpy as np


def normalizeFeatures(X, type="norm", min_vals=None, max_vals=None, mean_vals=None, std_vals=None):
    if type == "std":
        return (X - mean_vals)/std_vals
    elif type == "norm":
        return (X - min)/max - min
    elif type == "mean-norm":
        return (X - mean_vals)/max - min

        

def denormalizeFeatures(X, type="std", min_vals=None, max_vals=None, mean_vals=None, std_vals=None):
    if type == "std":
        return  X * std_vals + mean_vals
    elif type == "norm":
        return (X * (max - min)) + min
    elif type == "mean-norm":
        return (X * (max - min)) + mean_vals

def calculateMinMaxMeanStd(X):
    return np.min(X, axis=0), np.max(X, axis=0), np.mean(X, axis=0), np.std(X, axis=0) 
