from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np


def evaluate_model(Y, Y_pred):
    """Evaluate model performance."""
    mae = mean_absolute_error(Y, Y_pred)
    mse = mean_squared_error(Y, Y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(Y, Y_pred)

    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R² Score: {r2:.2f}")


def plot_results(X, Y, Y_pred):
    """Plot actual vs. predicted data."""

    plt.scatter(X, Y, color='red', label="Actual Data", marker='x')
    plt.plot(X, Y_pred, color='green', label="Best Fit Line")
    plt.xlabel("Features (X)")
    plt.ylabel("Output (Y)")
    plt.legend()
    plt.title("Linear Regression Model")
    plt.show()
