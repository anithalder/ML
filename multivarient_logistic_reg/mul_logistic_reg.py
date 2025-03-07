import numpy as np
import matplotlib.pyplot as plt


class MultivariateLogisticRegression:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.cost_history = []  # To store cost over iterations

    def _sigmoid(self, z):
        """Sigmoid function to map predictions to probabilities."""
        return 1 / (1 + np.exp(-z))

    def _compute_cost(self, y_true, y_pred):
        """Compute the binary cross-entropy cost function."""

        m = len(y_true)

        # Avoid division by zero and log(0) by adding a small value (epsilon)
        epsilon = 1e-15

        # Clip predictions to avoid log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        cost = -(1 / m) * np.sum(y_true * np.log(y_pred) +
                                 (1 - y_true) * np.log(1 - y_pred))
        return cost

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient Descent
        for _ in range(self.n_iters):
            # Linear model
            linear_model = np.dot(X, self.weights) + self.bias
            # Predictions using the sigmoid function
            y_pred = self._sigmoid(linear_model)

            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            # Update weights and bias
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            # Compute and store cost
            cost = self._compute_cost(y, y_pred)
            self.cost_history.append(cost)

    def predict(self, X, threshold=0.5):
        """Predict binary class labels."""
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = self._sigmoid(linear_model)
        return (y_pred >= threshold).astype(int)
