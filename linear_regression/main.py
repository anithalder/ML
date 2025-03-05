import numpy as np
import pandas as pd
from linear_regression import LinearRegression
from evaluation_visualization import evaluate_model, plot_results

df = pd.read_csv('data.csv')

X = np.array(df[['Features(X)']])
y = np.array(df[['Output(Y)']])

print(X.shape, y.shape)

# Initialize model
model = LinearRegression(learning_rate=0.01, epochs=1000)

# Train model
model.fit(X, y)

# Get predictions
Y_pred = model.predict(X)

# Get final parameters
m, c = model.get_params()
print(f"Final equation: y = {m:.2f}x + {c:.2f}")

# Plot the results
evaluate_model(y, Y_pred)
plot_results(X, y, Y_pred)
