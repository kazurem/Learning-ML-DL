import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from icecream import ic
import time
import ML.gradients
import ML.costs
plt.style.use("seaborn-v0_8")

'''
All of the functions assume that 2d matrices get passed . e.g (2, 1) instead of (2, ))
'''

def gradientDescentLinearRegression(X, y, w, b, num_iters=100, alpha=0.00001, return_cost_history=False, regul=False,regul_type="l2", lambda_=1, verbose=False):
    '''
    X --> (m x n)
    y --> (m x 1)
    W --> (n x 1)
    b --> int/float...
    '''

    w_init = w
    b_init = b

    if return_cost_history == True:
        cost_history = np.zeros((num_iters, 1))

    for i in range(num_iters):

        dj_dw, dj_db = ML.gradients.gradientLinearRegression(X, y, w_init, b_init, regul=regul, lambda_=lambda_, regul_type=regul_type)

        if np.linalg.norm(dj_dw) < 1e-6 and abs(dj_db) < 1e-6:
            break

        #increment w and b
        w_init = w_init - alpha*dj_dw
        b_init = b_init - alpha*dj_db

        if return_cost_history == True:
            cost_history[i][0] = ML.costs.costLinearRegression(X, y, w_init, b_init, regul=regul, lambda_=lambda_, regul_type=regul_type)

        if verbose == True and i % 100 == 0:
            print(f"W: {w_init}; b: {b_init}; cost: {cost_history[i][0]}; num_iters: {i}")

    if return_cost_history == True:
        return w_init, b_init, cost_history
    
    if w_init == np.nan or b_init == np.nan:
        print("Couldn't converge")

    return w_init, b_init


def gradientDescentLinearRegression(X, y, w, b, num_iters=100, alpha=0.00001, return_cost_history=False, regul=False,regul_type="l2", lambda_=1, verbose=False):
    '''
    X --> (m x n)
    y --> (m x 1)
    W --> (n x 1)
    b --> int/float...
    '''

    w_init = w
    b_init = b

    if return_cost_history == True:
        cost_history = np.zeros((num_iters, 1))

    for i in range(num_iters):

        dj_dw, dj_db = ML.gradients.gradientLinearRegression(X, y, w_init, b_init, regul=regul, lambda_=lambda_, regul_type=regul_type)

        if np.linalg.norm(dj_dw) < 1e-6 and abs(dj_db) < 1e-6:
            break

        #increment w and b
        w_init = w_init - alpha*dj_dw
        b_init = b_init - alpha*dj_db

        if return_cost_history == True:
            cost_history[i][0] = ML.costs.costLinearRegression(X, y, w_init, b_init, regul=regul, lambda_=lambda_, regul_type=regul_type)

        if verbose == True and i % 100 == 0:
            print(f"W: {w_init}; b: {b_init}; cost: {cost_history[i][0]}; num_iters: {i}")

    if return_cost_history == True:
        return w_init, b_init, cost_history
    
    if w_init == np.nan or b_init == np.nan:
        print("Couldn't converge")

    return w_init, b_init