import numpy as np
import ML.prediction_functions


def MeanSquaredError(X, y, w, b, regul=False, lambda_=1, regul_type="l2"):
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



def BinaryCrossEntropy(X, y, w, b, regul=False, lambda_=1, regul_type="l2"):

    total_cost = 0
    regul_cost = 0

    no_of_examples = X.shape[0]

    f_wb_i = ML.prediction_functions.sigmoid(np.matmul(X, w) + b)

    total_cost = -y * np.log(f_wb_i) - (1-y)*np.log(1-f_wb_i)
    total_cost = np.sum(total_cost)

    if regul == True:
        if regul_type == "l2":
            regul_cost = np.sum(np.square(w))
            regul_cost = (regul_cost) * (lambda_/(2*no_of_examples))
        elif regul_type == "l1":
            regul_cost = np.sum(np.abs(w))
            regul_cost = (regul_cost) * (lambda_/(2*no_of_examples))

    return total_cost + regul_cost


def distortion(X, closest_centroids, centroids, K):
    m = X.shape[0]
    cost = 0
    for i in range(K):
        grouped_points = X[closest_centroids.reshape(m) == i, :]
        if grouped_points.size > 0:
            cost += np.sum(np.sum(np.square(grouped_points - centroids[i]), axis=1))
    return cost/m


def SparseCrossCategoricalEntropy(X, y, w, b, regul=False, lambda_=1, regul_type="l2"):
    logit = ML.prediction_functions.linear(X, w, b, all_examples=True)
    softmaxed = ML.prediction_functions.softmax(logit)
    loss = -np.log(softmaxed)
