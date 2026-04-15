import numpy as np

# 1. Dataset & Setup
X = np.array([[1, 0, 1, 1, 0]], dtype=float).T # Shape (5, 1)
pos = np.arange(5).reshape(-1, 1) / 5
X_in = X + pos  # Add simple positional encoding

def softmax(x): 
    e = np.exp(x - np.max(x))
    return e / e.sum(axis=-1, keepdims=True)

def train_pixel_transformer(lr, epochs=10):
    # Simplified weights for Attention (Query, Key, Value) and Output
    Wq, Wk, Wv = [np.random.randn(1, 1) * 0.1 for _ in range(3)]
    Wo = np.random.randn(1, 1) * 0.1
    
    print(f"\nTraining with LR: {lr}")
    for epoch in range(1, epochs + 1):
        # 2. Scaled Dot-Product Attention
        Q, K, V = X_in @ Wq, X_in @ Wk, X_in @ Wv
        scores = (Q @ K.T) / np.sqrt(1)
        attn = softmax(scores)
        context = attn @ V
        
        # 3. Prediction & Loss (MSE)
        pred = context @ Wo
        loss = np.mean((X - pred)**2)
        
        # 4. Simple Gradient Descent (Manual Update)
        # Using a basic approximation for the gradient step
        grad = np.mean(2 * (pred - X))
        Wo -= lr * grad
        Wq -= lr * grad
        
        if epoch % 2 == 0: print(f"Epoch {epoch}, Loss: {loss:.4f}")
    return pred

# 5. Compare Learning Rates
pred_small = train_pixel_transformer(0.001)
pred_large = train_pixel_transformer(0.01)

print("\nNext Pixel Prediction (LR 0.01):", np.round(pred_large[-1], 2))