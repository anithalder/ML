import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score


def evaluate_model(Y, Y_pred):
    """Evaluate model performance."""
    mae = mean_absolute_error(Y, Y_pred)
    mse = mean_squared_error(Y, Y_pred)
    rmse = np.sqrt(mse)
    accuracy = accuracy_score(Y, Y_pred)

    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"Accuracy: {accuracy*100:.2f}%")
    # print(f"R² Score: {r2:.2f}")


def plot_results(X, Y, Y_pred):
    """Plot actual vs. predicted data."""
    # Ensure X and Y_pred have the same size
    if len(X) != len(Y_pred):
        raise ValueError("X and Y_pred must be the same size")

    plt.scatter(X, Y, color='red', label="Actual Data", marker='x')
    plt.plot(X, Y_pred, color='blue', label="Predicted Data")
    plt.scatter(X, Y_pred, color='blue', label="Predicted Data", marker='o')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()
