from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from dataset_load import load_dataset
from sklearn.model_selection import train_test_split

from evaluation_visualization import evaluate_model, plot_results
from logistic_reg import LogisticRegression

X, Y = load_dataset('student_study_data.csv')

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.20, random_state=42)

model = LogisticRegression(learning_rate=0.01, epoch=1000)
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)


evaluate_model(Y_test, Y_pred)


# Create a DataFrame with Y_test and Y_pred
results_df = pd.DataFrame({'Y_test': Y_test, 'Y_pred': Y_pred})

# Save the dataset to a CSV file
np.savetxt("predicted.csv", np.column_stack(
    (X_test, Y_pred)), delimiter=",", fmt="%.2f")

# Ensure X_test and Y_pred have the same size
if len(X_test) == len(Y_pred):
    plot_results(X_test, Y_test, Y_pred)
else:
    print("Error: X_test and Y_pred must be the same size")
