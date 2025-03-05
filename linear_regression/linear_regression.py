import numpy as np


class LinearRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.m = 0  # Initialize slope
        self.c = 0  # Initialize intercept

    def fit(self, X, Y):
        """Train the model using Gradient Descent with 1/2n factor in cost function."""
        n = len(X)  # Number of data points

        for _ in range(self.epochs):
            Y_pred = self.m * X + self.c  # Predictions
            error = Y - Y_pred  # Difference between actual and predicted values

            # Compute gradients with 1/n factor instead of 2/n
            dm = (-1 / n) * np.sum(X * error)
            dc = (-1 / n) * np.sum(error)

            # Update parameters
            self.m -= self.learning_rate * dm
            self.c -= self.learning_rate * dc

    def predict(self, X):
        """Predict output for given input X."""
        return self.m * X + self.c

    def cost_function(self, X, Y):
        """Compute the cost function (MSE with 1/2 factor)."""
        n = len(Y)
        Y_pred = self.predict(X)
        return (1 / (2 * n)) * np.sum((Y - Y_pred) ** 2)

    def get_params(self):
        return self.m, self.c
