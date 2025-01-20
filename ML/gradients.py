import functions
import numpy as np


def gradientLinearRegression(X: np.ndarray, y: np.ndarray, W: np.ndarray, b: np.float64):

    '''
    n --> number of features
    m --> number of training examples
    
    X --> (m x n)
    y --> (m x 1)
    W --> (n x 1)
    '''
    
    m, n = X.shape
    
    dj_dw = np.zeros((X.shape[1], 1))
    dj_db = 0

    predicted_y: np.ndarray = functions.linear(X, W, b)

    dj_dw = (predicted_y - y)
    dj_db = np.sum(predicted_y - y)


    for i in range(m):
        for j in range(m):
            dj_dw[j] = dj_dw[j] * X[i, j]

    return dj_dw, dj_db

    