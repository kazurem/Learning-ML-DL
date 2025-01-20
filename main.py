import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from icecream import ic
import time
import ML.models


df = pd.read_csv("./data/data.csv")

X_train = np.array(df['x']).reshape(-1, 1)
y_train = np.array(df['y']).reshape(-1, 1)

num_iters = 10000
w = np.zeros((X_train.shape[1], 1))
b = 0
cost_history = np.zeros((num_iters, 1))

start_time = time.time()
w, b = ML.models.gradientDescentLinearRegression(X_train, y_train, w, b, num_iters=num_iters)
end_time = time.time()

fig,ax = plt.subplots(1, 2, figsize=(20, 8))

ic(end_time - start_time)
ic(w, b)

ax[0].scatter(X_train, y_train, s=4)
ax[0].plot(X_train, X_train*w + b)
plt.show()