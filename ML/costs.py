import numpy as np


def costLinearRegression(X, y, w, b, regul=False, lambda_=1, regul_type="l2"):
    '''
    X --> (m x n)
    y --> (m x 1)
    W --> (n x 1)
    b --> int/float...
    '''

    total_cost = 0
    regul_cost = 0 #regularization cost

    no_of_examples = X.shape[0]
    total_cost = (np.matmul(X, w) + b) - y #calculate losses for all examples
    total_cost = np.square(total_cost)     #calculate the square of those losses
    total_cost = np.sum(total_cost)        #sum all the squared losses
    total_cost = total_cost/(2*no_of_examples) #get the total cost

    if regul == True:
        if regul_type == "l2":
            regul_cost = np.sum(np.square(w))
            regul_cost = (regul_cost) * (lambda_/(2*no_of_examples))
        elif regul_type == "l1":
            regul_cost = np.sum(np.abs(w))
            regul_cost = (regul_cost) * (lambda_/(2*no_of_examples))

    return total_cost + regul_cost

