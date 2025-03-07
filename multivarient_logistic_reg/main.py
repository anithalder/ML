# Description: Main file to run the logistic regression model on a sample dataset.
from dataset_load import load_data
from mul_logistic_reg import MultivariateLogisticRegression
from evaluation_visulization import evaluate_model, plot_cost


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
    X, y = load_data("dataset_mul.csv")

    # Create and train model
    model = MultivariateLogisticRegression(learning_rate=0.01, n_iters=1000)
    model.fit(X, y)

    # Make predictions
    y_pred = model.predict(X)

    print("Predicted values:", y_pred[:5])

    evaluate_model(y, y_pred)

    # Plot cost over iterations
    plot_cost(model)
