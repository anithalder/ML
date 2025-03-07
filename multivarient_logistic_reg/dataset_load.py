import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(filename="dataset.csv"):
    """Load dataset from CSV file."""
    data = pd.read_csv(filename)

    # Extract all features (except last column)
    X = data.drop(columns=data.columns[-1]).values

    # Extract last column as output
    Y = data[data.columns[-1]].values
    return X, Y
