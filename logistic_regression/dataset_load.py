import pandas as pd


def load_dataset(filename='dataset.csv'):
    # Load the dataset
    """Load dataset from CSV file."""
    data = pd.read_csv(filename)
    X = data.iloc[:, :-2]  # Single feature
    Y = data.iloc[:, -1]  # Binary target
    return X, Y
