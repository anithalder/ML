

import numpy as np
from sklearn import datasets


def load_dataset():

    iris = datasets.load_iris()
    X = iris.data[:, :2]  # Use only the first two features for simplicity
    y = iris.target
    y = np.where(y == 0, -1, 1)

    # Load the dataset from the CSV file
    # df = pd.read_csv(filename)

    # # Extract the features and labels
    # X = df.iloc[:, :-1].values
    # y = df.iloc[:, -1].values
    # y = np.where(y == 0, -1, 1)

    return X, y
