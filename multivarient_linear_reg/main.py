import numpy as np
from dataset_load import load_data
from mul_linear_reg import MultivariateLogisticRegression


if __name__ == "__main__":
    # Generate a sample binary classification dataset
    # np.random.seed(42)
    # X = np.random.randn(100, 3)  # 100 samples, 3 features
    # Binary target (0 or 1)
    # y = (X[:, 0] + X[:, 1] + X[:, 2] > 0).astype(int)

    # Save the dataset to a CSV file
    # np.savetxt("dataset.csv", np.column_stack(
    #     (X, y)), delimiter=",", fmt="%.2f")

    # Load the dataset
    X, y = load_data("dataset.csv")

    # Create and train model
    model = MultivariateLogisticRegression(learning_rate=0.01, n_iters=1000)
    model.fit(X, y)

    # Make predictions
    y_pred = model.predict(X)

    # Evaluate model
    accuracy = model.accuracy(y, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # Plot cost over iterations
    model.plot_cost()
