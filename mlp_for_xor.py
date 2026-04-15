import numpy as np

# XOR Dataset
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])

def sigmoid(x): return 1 / (1 + np.exp(-x))
def sigmoid_der(x): return x * (1 - x)

def train_xor(lr, epochs=10):
    np.random.seed(42)
    # Init weights: Input(2) -> Hidden(2) -> Output(1)
    wh = np.random.uniform(size=(2, 2))
    wo = np.random.uniform(size=(2, 1))
    
    print(f"\nTraining with Learning Rate: {lr}")
    for i in range(epochs):
        # Forward
        hi = sigmoid(np.dot(X, wh))
        out = sigmoid(np.dot(hi, wo))
        
        # Loss (MSE)
        loss = np.mean((y - out)**2)
        
        # Backprop
        d_out = (y - out) * sigmoid_der(out)
        d_hi = d_out.dot(wo.T) * sigmoid_der(hi)
        
        # Update
        wo += hi.T.dot(d_out) * lr
        wh += X.T.dot(d_hi) * lr
        
        if (i+1) % 2 == 0: print(f"Epoch {i+1}, Loss: {loss:.4f}")
    return out

# Compare Learning Rates
pred_01 = train_xor(0.01)
pred_1 = train_xor(0.1)

print("\nFinal Predictions (LR 0.1):")
print(np.round(pred_1, 2))