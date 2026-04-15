import numpy as np

# 1. Dataset (Digit 0 and Digit 1)
d0 = np.array([1,1,1,1,1, 1,0,0,0,1, 1,0,0,0,1, 1,0,0,0,1, 1,1,1,1,1]).reshape(5,5)
d1 = np.array([0,0,1,0,0, 0,1,1,0,0, 1,0,1,0,0, 0,0,1,0,0, 1,1,1,1,1]).reshape(5,5)
X = [d0, d1]
y = [0, 1]  # Labels

def pool(img): # 2x2 Max Pooling on 4x4 conv output
    res = np.zeros((2,2))
    for i in range(2): 
        for j in range(2): res[i,j] = np.max(img[i*2:i*2+2, j*2:j*2+2])
    return res

def train_digit_cnn(lr, epochs=10):
    np.random.seed(42)
    filt = np.random.randn(2,2) * 0.5
    w_out = np.random.randn(4) * 0.5 # 2x2 pooled map flattened

    print(f"\nTraining with LR: {lr}")
    for ep in range(1, epochs+1):
        total_loss = 0
        for img, target in zip(X, y):
            # Conv (4x4) -> Pool (2x2)
            conv = np.array([[np.sum(img[i:i+2, j:j+2]*filt) for j in range(4)] for i in range(4)])
            p_out = pool(np.maximum(0, conv)) # ReLU
            pred = 1 / (1 + np.exp(-np.sum(p_out.flatten() * w_out)))
            
            # Update (Simplified)
            err = pred - target
            w_out -= lr * err * p_out.flatten()
            filt -= lr * err * np.mean(img[:2,:2])
            total_loss += err**2
        if ep % 5 == 0: print(f"Epoch {ep}, Loss: {total_loss/2:.4f}")
    return filt, w_out

# 2. Execution
f, w = train_digit_cnn(0.1)

# 3. Classify New Pattern (Modified Digit 1)
new_test = np.array([0,0,1,0,0, 0,0,1,0,0, 0,0,1,0,0, 0,0,1,0,0, 0,0,1,0,0]).reshape(5,5)
c_ = np.array([[np.sum(new_test[i:i+2, j:j+2]*f) for j in range(4)] for i in range(4)])
p_ = pool(np.maximum(0, c_))
final_pred = 1 / (1 + np.exp(-np.sum(p_.flatten() * w)))

print(f"\nNew Pattern Probability (Target 1): {final_pred:.4f}")
print("Classification:", "Digit 1" if final_pred > 0.5 else "Digit 0")