import numpy as np

# 1. Dataset (3x3 patterns flattened to 9D)
clean_img = np.array([1, 1, 1, 1, 0, 1, 1, 1, 1], dtype=float).reshape(1, 9)
noisy_img = np.array([1, 0, 1, 1, 1, 1, 1, 0, 1], dtype=float).reshape(1, 9)

def sigmoid(x): return 1 / (1 + np.exp(-x))
def sigmoid_der(x): return x * (1 - x)

def train_autoencoder(lr, epochs=10):
    np.random.seed(42)
    # Weights: Input(9) -> Hidden(4) -> Output(9)
    W1 = np.random.randn(9, 4) * 0.5
    W2 = np.random.randn(4, 9) * 0.5
    
    print(f"\nTraining Autoencoder with LR: {lr}")
    losses = []
    for epoch in range(1, epochs + 1):
        # Forward Pass
        hidden = sigmoid(noisy_img @ W1)
        output = sigmoid(hidden @ W2)
        
        # Loss (MSE)
        loss = np.mean((output - clean_img)**2)
        losses.append(loss)
        
        # Backprop
        error_out = (output - clean_img) * sigmoid_der(output)
        error_hidden = (error_out @ W2.T) * sigmoid_der(hidden)
        
        # Update
        W2 -= lr * (hidden.T @ error_out)
        W1 -= lr * (noisy_img.T @ error_hidden)
        
        if epoch % 2 == 0: print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    return output, losses[-1]

# 2. Execute with different Learning Rates
pred_small, loss_small = train_autoencoder(0.001)
pred_large, loss_large = train_autoencoder(0.01)

# 3. Results
print("\n" + "="*30)
print("EVALUATION")
print("="*30)
print(f"Loss Difference (LR 0.001 - LR 0.01): {loss_small - loss_large:.6f}")
print("\nOriginal Clean Image:", clean_img.flatten().astype(int))
print("Denoised Image (LR 0.01):", np.round(pred_large).flatten().astype(int))