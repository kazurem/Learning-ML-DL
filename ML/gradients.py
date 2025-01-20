import numpy as np
import ML.prediction_functions

def gradientLinearRegression(X, y, w, b, regul=False, lambda_=1, regul_type="l2"):

    '''
    X --> (m x n)
    y --> (m x 1)
    W --> (n x 1)
    b --> int/float...
    '''

    no_of_examples, no_of_features = X.shape 


    dj_dw = np.zeros((no_of_features, 1)) #gradient of cost function wrt w (weights)
    dj_db = 0 #gradient of cost function wrt b (bias)

    err = (np.matmul(X, w) + b) - y 

    dj_dw_tmp = np.sum(X*err, axis=0) 
    dj_dw = dj_dw_tmp.T

    dj_db = np.sum(err)

    dj_dw = dj_dw/no_of_examples
    dj_db = dj_db/no_of_examples

    if regul == True:
        if regul_type == "l2":
            dj_dw = dj_dw + ((lambda_/no_of_examples) * w)

    return dj_dw, dj_db


def gradientLogisticRegression(X, y, w, b, regul=False, lambda_=1, regul_type="l2"):

    '''
    X --> (m x n)
    y --> (m x 1)
    W --> (n x 1)
    b --> int/float...
    '''

    no_of_examples, no_of_features = X.shape 


    dj_dw = np.zeros((no_of_features, 1)) #gradient of cost function wrt w (weights)
    dj_db = 0 #gradient of cost function wrt b (bias)

    err = ML.prediction_functions.sigmoid(np.matmul(X, w) + b) - y 

    dj_dw_tmp = np.sum(X*err, axis=0) 
    dj_dw = dj_dw_tmp.T

    dj_db = np.sum(err)

    dj_dw = dj_dw/no_of_examples
    dj_db = dj_db/no_of_examples

    if regul == True:
        if regul_type == "l2":
            dj_dw = dj_dw + ((lambda_/no_of_examples) * w)

    return dj_dw, dj_db



