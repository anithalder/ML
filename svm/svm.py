import numpy as np
import matplotlib.pyplot as plt


class SVM_C:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        y_ = np.where(y <= 0, -1, 1)

        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * \
                        (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]

    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)

    def visualize(self, X, y, y_pred):
        def get_hyperplane_value(x, w, b, offset):
            return (-w[0] * x + b + offset) / w[1]

        plt.scatter(X[:, 0], X[:, 1], c=y, cmap='winter')
        ax = plt.gca()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        xx = np.linspace(xlim[0], xlim[1])
        yy = np.linspace(ylim[0], ylim[1])
        YY, XX = np.meshgrid(yy, xx)
        xy = np.vstack([XX.ravel(), YY.ravel()]).T
        Z = self.get_hyperplane(xy).reshape(XX.shape)
        ax.contour(XX, YY, Z, colors='k',
                   levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])

        plt.show()

    def get_params(self):
        return self.w, self.b

    def get_hyperplane(self, X):
        return np.dot(X, self.w) - self.b
