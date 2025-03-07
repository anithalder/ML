import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score


def evaluate_model(Y, Y_pred):
    """Evaluate model performance."""
    # Evaluate model
    accuracy = accuracy_score(Y, Y_pred)
    mse = mean_squared_error(Y, Y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(Y, Y_pred)
    r2 = r2_score(Y, Y_pred)

    # Evaluation print statements
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"Root Mean Squared Error: {rmse:.2f}")
    print(f"R-squared: {r2:.2f}")
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"Accuracy: {accuracy * 100:.2f}%")


def plot_cost(self):
    """Plot the cost over iterations."""
    plt.plot(range(self.n_iters), self.cost_history)
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.title("Cost Over Iterations")
    plt.show()
