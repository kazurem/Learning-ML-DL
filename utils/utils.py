import time
import numpy as np

def timer(func):

    def wrapper(*args, **kwargs):
        start_time = time.time()
        w, b, cost_history = func(*args, **kwargs)
        end_time = time.time()
        print(f"Time taken: {end_time - start_time}")
        return w, b, cost_history

    return wrapper

def multivariate_gaussian(X, mean, std):
    p_x = (1/(np.sqrt(2*np.pi)*std)) * np.exp((-np.square(X-mean))/(2*np.square(std)))
    return p_x[:, 0] * p_x[:, 1]