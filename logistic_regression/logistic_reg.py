import numpy as np


class LogisticRegression:

    def __init__(self, learning_rate=0.01, epoch=1000):
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.W = None

    def fit(self, X, Y):
        # Initialize weights
        # Change to X.shape[1] to match the number of features
        self.W = np.zeros(X.shape[1])
        self.b = 0
        m = len(X)

        for _ in range(self.epoch):
            z = np.dot(X, self.W) + self.b
            Y_pred = self.sigmoid(z)

            dw = (1 / m) * np.dot(X.T, (Y_pred - Y))
            db = (1 / m) * np.sum(Y_pred - Y)

            self.W -= self.learning_rate * dw
            self.b -= self.learning_rate * db

            if _ % 100 == 0:
                print(
                    f'Iteration {_} cost {self.cost_function(Y, Y_pred):.2f}')

    def sigmoid(self, z):
        """Compute the sigmoid function."""
        return (1 / (1 + np.exp(-z)))

    def cost_function(self, Y, Y_pred):
        """Compute the binary cross-entropy loss."""
        n = len(Y)
        epsilon = 1e-15  # Add epsilon to avoid log(0)
        return float((-1/n) * np.sum(Y * np.log(Y_pred + epsilon) + (1 - Y) * np.log(1 - Y_pred + epsilon)))

    def predict(self, X):
        z = np.dot(X, self.W) + self.b
        Y_pred = self.sigmoid(z)
        return np.round(Y_pred)
