import numpy as np
import matplotlib.pyplot as plt

# 1. Dataset
X = np.array([500, 800, 1000, 1200, 1500], dtype=float)
Y = np.array([150, 220, 300, 360, 450], dtype=float)

# Normalize data for better convergence
X_mean, X_std = np.mean(X), np.std(X)
X_scaled = (X - X_mean) / X_std

# 2. Initialize weights and bias
weight = 0.0
bias = 0.0
learning_rate = 0.01
epochs = 1000
losses = []

# 3. Training with Gradient Descent
for epoch in range(epochs):
    # Forward Pass
    Y_pred = weight * X_scaled + bias
    
    # Calculate Mean Squared Error Loss
    loss = np.mean((Y_pred - Y)**2)
    losses.append(loss)
    
    # Backward Pass (Gradients)
    dw = (2/len(X)) * np.sum((Y_pred - Y) * X_scaled)
    db = (2/len(X)) * np.sum(Y_pred - Y)
    
    # Update parameters
    weight -= learning_rate * dw
    bias -= learning_rate * db

print(f'Final Weight: {weight:.4f}, Final Bias: {bias:.4f}')
# 4. Show loss reduction
plt.figure(figsize=(10, 4))
plt.plot(losses)
plt.title('Loss Reduction over Epochs')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.grid(True)
plt.show()
# 5. Predict price for 1100 sq.ft
test_area = 1100
test_area_scaled = (test_area - X_mean) / X_std
predicted_price = weight * test_area_scaled + bias

print(f'Predicted price for {test_area} sq.ft: ${predicted_price:.2f}k')
# 6. Plot regression line
plt.scatter(X, Y, color='blue', label='Actual Data')
full_x_range = np.linspace(min(X), max(X), 100)
full_x_scaled = (full_x_range - X_mean) / X_std
full_y_pred = weight * full_x_scaled + bias

plt.plot(full_x_range, full_y_pred, color='red', label='Regression Line')
plt.xlabel('Area (sq.ft)')
plt.ylabel('Price ($k)')
plt.title('House Price Prediction')
plt.legend()
plt.grid(True)
plt.show()