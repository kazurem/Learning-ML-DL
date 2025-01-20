import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import costs
import gradients
plt.style.use("seaborn-v0_8")


def gradientDescentLinearRegression(X: np.ndarray, y: np.ndarray, W: np.ndarray, b: np.float64, num_iters: int=100, alpha: float = 0.01, regul=False, lambda_=1):

    dj_dw = np.zeros((X.shape[1], 1))
    dj_db = 0

    w_init = W
    b_init = b

    cost_history: np.float64 = np.zeros((num_iters, 1));

    for i in range(num_iters):
        dj_dw, dj_db = gradients.gradientLinearRegression(X, y, W, b)

        w_init = w_init - alpha*dj_dw
        b_init = b_init - alpha*dj_db

        cost_history[i] = costs.MeanSquaredError(X, y, W, b, regul=regul, lambda_=lambda_)

        if i % 10 == 0:
            print(f"W: {w_init}; b: {b_init}; cost: {cost_history[i]}")

    return w_init, b_init,cost_history