import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("data (1).csv")

X_train = np.array(df['x']).reshape(-1, 1)
X_train1 = np.array(df['x']).reshape(-1, 1)
X_train3 = np.hstack((X_train, X_train1))
y_train = np.array(df['y']).reshape(-1, 1)

print(X_train3)
print(np.std(X_train3, axis=0))

# plt.scatter(df['x'], df['y'])
# x = np.arange(500)
# plt.plot(x, np.full(500, np.mean(X_train)))
# plt.show()
