import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data.csv')

X = np.array(df[['X']])
y = np.array(df[['Y']])

plt.scatter(X, y, color="blue", label="Sample Data")  # Plot all data points
plt.xlabel("X")
plt.ylabel("y")
plt.title("Sample Dataset (Before Model Training)")
plt.legend()
plt.show()
