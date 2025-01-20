import numpy as np

def linear(X: np.ndarray, W: np.ndarray, b: np.float64) -> np.float64:
    return (np.matmul(X, W) + b)

def sigmoid(z: np.ndarray):
    return 1/(1 + np.exp(-z))