from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def evaluate_model(y_true, y_pred, average='binary'):
    """
    Evaluate the SVM model using accuracy, precision, recall, and F1-score.

    Parameters:
    y_true (array-like): True labels.
    y_pred (array-like): Predicted labels.
    average (str): The type of averaging to be performed on the data.

    Returns:
    dict: A dictionary containing accuracy, precision, recall, and F1-score.
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(
        y_true, y_pred, average=average, zero_division=1)
    recall = recall_score(y_true, y_pred, average=average, zero_division=1)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=1)

    return {
        "Accuracy": accuracy*100,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1
    }
