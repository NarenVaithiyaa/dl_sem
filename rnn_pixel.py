import numpy as np

# 1. Dataset (Sequence of 3 rows, each 3 pixels)
X_seq = np.array([
    [1, 1, 1], # Row 1
    [0, 1, 0], # Row 2
    [1, 1, 1]  # Row 3
], dtype=float)

# Target: 1 (Pattern matches a specific class)
y_target = 1.0

def sigmoid(x): return 1 / (1 + np.exp(-x))
def sigmoid_der(x): return x * (1 - x)

def train_rnn(lr, epochs=10):
    np.random.seed(42)
    # Weights: Input to Hidden (3x2), Hidden to Hidden (2x2), Hidden to Output (2x1)
    Wxh = np.random.randn(3, 2) * 0.5
    Whh = np.random.randn(2, 2) * 0.5
    Why = np.random.randn(2, 1) * 0.5
    
    print(f"\nTraining RNN with LR: {lr}")
    for epoch in range(1, epochs + 1):
        # Forward Pass
        h = np.zeros((1, 2))
        for row in X_seq:
            h = np.tanh(row.reshape(1, 3) @ Wxh + h @ Whh)
            
        y_pred = sigmoid(h @ Why)
        loss = np.mean((y_pred - y_target)**2)
        
        # Backpropagation (Gradient Descent)
        # Error at the output
        error_out = (y_pred - y_target) * sigmoid_der(y_pred)
        
        # Gradient for Why
        d_Why = h.T @ error_out
        
        # Gradient for hidden state (tanh derivative is 1 - h^2)
        d_h = (error_out @ Why.T) * (1 - h**2)
        
        # Gradient for Wxh (using the last input row for simplicity in this toy example)
        d_Wxh = X_seq[-1:].T @ d_h
        
        # Update weights
        Why -= lr * d_Why
        Wxh -= lr * d_Wxh

        if epoch % 2 == 0: 
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
            
    return y_pred[0][0]

# 2. Compare Learning Rates
pred_01 = train_rnn(0.01)
pred_1 = train_rnn(0.1)

print(f"\nFinal Prediction (LR 0.1): {pred_1:.4f}")
print("Pattern Class:", "Class A" if pred_1 > 0.5 else "Class B")