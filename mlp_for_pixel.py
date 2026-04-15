import numpy as np

# 1. Dataset (2x2 images flattened to 4D)
# [255, 255, 255, 255] -> Bright (1), [10, 10, 10, 10] -> Dark (0)
X = np.array([[255, 255, 255, 255], [10, 10, 10, 10]], dtype=float)
y = np.array([[1], [0]], dtype=float)

# 2. Normalize Pixel Values (0-1 range)
X_norm = X / 255.0

def sigmoid(x): return 1 / (1 + np.exp(-x))
def sigmoid_der(x): return x * (1 - x)

def train_mlp(lr, epochs=10):
    np.random.seed(42)
    # Weights: Input(4) -> Hidden(3) -> Output(1)
    W1 = np.random.randn(4, 3) * 0.5
    W2 = np.random.randn(3, 1) * 0.5
    
    print(f"\nTraining MLP with LR: {lr}")
    for epoch in range(1, epochs + 1):
        # Forward Pass
        h_layer = sigmoid(X_norm @ W1)
        output = sigmoid(h_layer @ W2)
        
        # Loss (MSE)
        loss = np.mean((y - output)**2)
        
        # Backprop
        error_out = (output - y) * sigmoid_der(output)
        error_h = (error_out @ W2.T) * sigmoid_der(h_layer)
        
        # Update
        W2 -= lr * (h_layer.T @ error_out)
        W1 -= lr * (X_norm.T @ error_h)
        
        if epoch % 2 == 0: print(f"Epoch {epoch}, Loss: {loss:.4f}")
    return output, W1, W2

# 3. Train and Compare
out_small, _, _ = train_mlp(0.001)
out_large, final_W1, final_W2 = train_mlp(0.01)

# 4. Predict for New Pattern (e.g., [120, 120, 120, 120])
new_img = np.array([[120, 120, 120, 120]]) / 255.0
h_new = sigmoid(new_img @ final_W1)
pred_new = sigmoid(h_new @ final_W2)

print("\n" + "="*30)
print("RESULTS")
print("="*30)
print(f"Prediction for [120, 120, 120, 120]: {pred_new[0][0]:.4f} ({'Bright' if pred_new > 0.5 else 'Dark'})")
print("Decision Boundary Note: The boundary exists where the hidden-to-output weighted sum is 0 (sigmoid=0.5).")