import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from icecream import ic
# from ML.models import gradientDescentLogisticRegression
from ML import prediction_functions
from ML.models import KNN
from ML.models import LinearRegressor, LogisticRegressor
import time
plt.style.use("seaborn-v0_8")

# df = pd.read_csv("./data/data.csv")

# X_train = np.array(df['x']).reshape(-1, 1)
# y_train = np.array(df['y']).reshape(-1, 1)

# num_iters = 10000
# w = np.zeros((X_train.shape[1], 1))
# b = 0
# cost_history = np.zeros((num_iters, 1))

# model = LinearRegressor(num_iters=num_iters)

# start_time = time.time()
# w, b, cost_history = model.fit(X_train, y_train, verbose=True, return_cost_history=True)
# end_time = time.time()

# fig,ax = plt.subplots(1, 2, figsize=(20, 8))

# ic(end_time - start_time)
# ic(w, b)

# ax[0].scatter(X_train, y_train, s=4)
# ax[0].plot(X_train, X_train*w + b)
# ax[1].plot(np.arange(num_iters), cost_history)
# plt.show()

# X_train = np.array([[1], [2], [3], [4], [5], [6]])
# y_train = np.array([[0], [0], [0], [1], [1], [1]])
# w = np.zeros((X_train.shape[1], 1))
# b = 0
# num_iters = 5000
# cost_history = np.zeros((num_iters, 1))


# model = LogisticRegressor(num_iters=num_iters, learning_rate=0.5)

# w, b, cost_history = model.fit(X_train, y_train, return_cost_history=True, verbose=True)




# fig, ax = plt.subplots(1, 2, figsize=(15, 6))

# ic(prediction_functions.sigmoid(X_train*w + b))


# ax[0].scatter(X_train, y_train)
# ax[0].plot(X_train, prediction_functions.sigmoid(X_train*w + b), c="black")
# ax[1].plot(np.arange(num_iters), cost_history)
# plt.show()  

df = pd.read_csv("data/data (1).csv")
model = KNN(no_of_neighbors=10)
model.fit(df.iloc[:, :2], df['z'])
print(model.X)
x_coord = 100
y_coord = 150
X = np.array([[150, 100]])
pred = model.predict(X)
print(pred)
pos = df[df['z'] == 'a']
neg = df[df['z'] == 'b']
plt.scatter(pos['x'], pos['y'])
plt.scatter(neg['x'], neg['y'], c="black")
plt.scatter(X[:, 0], X[:, 1], c="red")
plt.show()