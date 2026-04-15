import numpy as np
import matplotlib.pyplot as plt

# 1. Dataset (4x4 grid with a cross pattern)
X = np.array([
    [0, 1, 0, 0],
    [1, 1, 1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 0]
], dtype=float)
y_target = 1.0

def relu(x): return np.maximum(0, x)

def train_cnn(lr, epochs=10):
    np.random.seed(42)
    # 2x2 Filter and Output weight
    filt = np.random.randn(2, 2) * 0.5
    w_out = np.random.randn(9) * 0.5 # 3x3 activation map flattened
    
    print(f"\nTraining CNN with LR: {lr}")
    for epoch in range(1, epochs + 1):
        # 2. Convolution Step (Stride=1, No Padding)
        # Input 4x4 -> Filter 2x2 -> Output 3x3
        output_map = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                output_map[i, j] = np.sum(X[i:i+2, j:j+2] * filt)
        
        # 3. Activation & Prediction
        activation = relu(output_map)
        pred = np.sum(activation.flatten() * w_out)
        loss = (pred - y_target)**2

        # 4. Manual Gradient Descent (Simplified)
        error = 2 * (pred - y_target)
        w_out -= lr * error * activation.flatten()
        # Simple update for filter based on total error
        filt -= lr * error * np.mean(X[:2, :2]) 

        if epoch % 2 == 0: print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    return pred, activation

# 5. Execute and Visualize
pred_01, act_01 = train_cnn(0.01)
pred_1, act_1 = train_cnn(0.1)

print(f"\nFinal Prediction (LR 0.1): {pred_1:.4f}")

plt.figure(figsize=(8, 3))
plt.subplot(1, 2, 1)
plt.title("Input Pattern")
plt.imshow(X, cmap='binary')
plt.subplot(1, 2, 2)
plt.title("Activation Map (LR 0.1)")
plt.imshow(act_1, cmap='viridis')
plt.colorbar()
plt.show()