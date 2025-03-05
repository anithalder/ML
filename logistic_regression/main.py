import pandas as pd
import numpy as np
from dataset_load import load_dataset
from sklearn.model_selection import train_test_split

from evaluation_visualization import evaluate_model, plot_results
from logistic_reg import LogisticRegression

X, Y = load_dataset('dataset.csv')

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.20, random_state=42)

model = LogisticRegression(learning_rate=0.01, epoch=1000)
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)

evaluate_model(Y_test, Y_pred)
# plot_results(X_test, Y_test, Y_pred)
accuracy = np.mean(Y_test == Y_pred)
print(f"Accuracy: {accuracy}")

# Create a DataFrame with Y_test and Y_pred
results_df = pd.DataFrame({'Y_test': Y_test, 'Y_pred': Y_pred})

# Print the DataFrame
print(results_df.head(10))
