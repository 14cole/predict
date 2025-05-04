import numpy as np

# --- 1. INPUT DATA ---
# Example: 5 samples, 3 variables (you can load this from CSV or user input)
X = np.array([
    [1.2, 3.4, 5.6],
    [2.1, 0.4, 3.3],
    [3.3, 2.2, 1.1],
    [4.4, 5.5, 2.2],
    [5.5, 1.1, 4.4]
])

# Target variable (length must match number of rows in X)
y = np.array([10.0, 12.5, 13.2, 15.5, 18.1])

# --- 2. ADD INTERCEPT TERM ---
# Adds a column of 1s to X for the intercept (bias)
X_bias = np.hstack([np.ones((X.shape[0], 1)), X])

# --- 3. COMPUTE REGRESSION COEFFICIENTS ---
# θ = (XᵗX)⁻¹ Xᵗy
theta = np.linalg.inv(X_bias.T @ X_bias) @ X_bias.T @ y

# --- 4. DEFINE PREDICTION FUNCTION ---
def predict(X_new):
    X_new = np.array(X_new)
    if X_new.ndim == 1:
        X_new = X_new.reshape(1, -1)
    X_new_bias = np.hstack([np.ones((X_new.shape[0], 1)), X_new])
    return X_new_bias @ theta

# --- 5. EXAMPLE PREDICTION ---
new_data = [
    [2.0, 3.0, 4.0],
    [5.0, 2.0, 1.0]
]
predictions = predict(new_data)

# --- 6. OUTPUT ---
print("Regression Coefficients (theta):", theta)
for i, pred in enumerate(predictions):
    print(f"Prediction for input {new_data[i]} → {pred:.2f}")
