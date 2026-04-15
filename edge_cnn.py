import numpy as np

# 1. Dataset (3x3 grid with horizontal edges)
X = np.array([
    [10, 10, 10],
    [ 0,  0,  0],
    [10, 10, 10]
], dtype=float)
y_target = 1.0 # Target: detection of high contrast

def train_edge_cnn(lr, epochs=10):
    np.random.seed(42)
    # 2x2 kernel (Filter)
    kernel = np.random.randn(2, 2) * 0.1
    
    print(f"\nTraining Edge CNN with LR: {lr}")
    for epoch in range(1, epochs + 1):
        # 2. Convolution (Stride=1, No Padding) -> 2x2 Output
        feat_map = np.zeros((2, 2))
        for i in range(2):
            for j in range(2):
                feat_map[i, j] = np.sum(X[i:i+2, j:j+2] * kernel)
        
        # 3. Prediction (Mean of feature map)
        pred = np.mean(feat_map)
        loss = (pred - y_target)**2
        
        # 4. Manual Gradient Descent
        grad = 2 * (pred - y_target)
        kernel -= lr * grad * np.mean(X[:2, :2])
        
        if epoch % 2 == 0: print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    return feat_map, kernel

# 5. Compare Learning Rates
map_small, kern_small = train_edge_cnn(0.01)
map_large, kern_large = train_edge_cnn(0.5)

print("\n" + "="*30)
print("FEATURE MAP INTERPRETATION")
print("="*30)
print("Final Kernel (LR 0.5):\n", np.round(kern_large, 2))
print("\nFinal Feature Map (Edges highlighted):\n", np.round(map_large, 2))
print("Interpretation: High values in the feature map indicate where the horizontal edge was detected.")