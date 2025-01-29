import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from icecream import ic
from ML import prediction_functions
from ML.supervised import KNN
from ML.supervised import LinearRegressor, LogisticRegressor
from ML.unsupervised import KMeans, AnomalyDetection
from utils.utils import multivariate_gaussian
from sklearn.datasets import make_blobs
import time
plt.style.use("seaborn-v0_8")



#Anomaly Detection
df = pd.read_csv("data/anomaly.csv")

model = AnomalyDetection()

X = (df.iloc[:, :2]).to_numpy()
model.fit(X)

y_pred = model.predict(X, epsilon=1e-7)
X_anomalous = X[y_pred]
X_non_anomalous = X[~y_pred]
plt.scatter(X_non_anomalous[:, 0], X_non_anomalous[:, 1], s=20)
plt.scatter(X_anomalous[:, 0], X_anomalous[:, 1], s=20, c="red", marker="x")

plt.show()

