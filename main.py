import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from icecream import ic
from ML import prediction_functions
from ML import models
import time
plt.style.use("seaborn-v0_8")

df = pd.read_csv("./data/data.csv")

X_train = np.array(df['x']).reshape(-1, 1)
y_train = np.array(df['y']).reshape(-1, 1)

num_iters = 1000
w = np.zeros((X_train.shape[1], 1))
b = 0
cost_history = np.zeros((num_iters, 1))

start_time = time.time()
w, b = models.gradientDescentLinearRegression(X_train, y_train, w, b, num_iters=num_iters)
end_time = time.time()

fig,ax = plt.subplots(1, 2, figsize=(20, 8))

ic(end_time - start_time)
ic(w, b)

ax[0].scatter(X_train, y_train, s=4)
ax[0].plot(X_train, X_train*w + b)
plt.show()

# X_train = np.array([[1], [2], [3], [4], [5], [6]])
# y_train = np.array([[0], [0], [0], [1], [1], [1]])
# w = np.zeros((X_train.shape[1], 1))
# b = 0
# num_iters = 5000
# cost_history = np.zeros((num_iters, 1))


# w, b, cost_history = models.gradientDescentLogisticRegression(X_train, y_train, w, b, num_iters=num_iters, return_cost_history=True, alpha=0.4)


# fig, ax = plt.subplots(1, 2, figsize=(15, 6))

# ax[0].scatter(X_train, y_train)
# ax[0].plot(X_train, prediction_functions.sigmoid(X_train*w + b), c="black")
# ax[1].plot(np.arange(num_iters), cost_history)
# plt.show()