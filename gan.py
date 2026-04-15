import numpy as np

# Dataset
real_data = np.array([[1, 0, 1, 0], 
                      [0, 1, 0, 1]])

def sigmoid(x): 
    return 1 / (1 + np.exp(-x))

def binary_cross_entropy(y_true, y_pred):
    eps = 1e-8
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def train_gan_corrected(lr, epochs=10):
    np.random.seed(42)
    
    # Generator: 2D noise → 4D pattern
    Wg = np.random.randn(2, 4) * 0.5
    bg = np.zeros((1, 4))
    
    # Discriminator: 4D pattern → 1D validity
    Wd = np.random.randn(4, 1) * 0.5
    bd = np.zeros((1, 1))
    
    print(f"\nTraining GAN with LR: {lr}")
    
    for epoch in range(1, epochs + 1):
        # ----- Train Discriminator -----
        # Real data should be classified as 1
        real_score = sigmoid(real_data @ Wd + bd)
        d_real_loss = binary_cross_entropy(np.ones_like(real_score), real_score)
        
        # Fake data should be classified as 0
        noise = np.random.randn(len(real_data), 2)
        fake_data = sigmoid(noise @ Wg + bg)
        fake_score = sigmoid(fake_data @ Wd + bd)
        d_fake_loss = binary_cross_entropy(np.zeros_like(fake_score), fake_score)
        
        d_loss = (d_real_loss + d_fake_loss) / 2
        
        # Simple gradient update (simplified backprop)
        grad_real = real_data.T @ (real_score - 1)
        grad_fake = fake_data.T @ fake_score
        Wd -= lr * (grad_real + grad_fake) / len(real_data)
        bd -= lr * np.mean((real_score - 1) + fake_score, axis=0, keepdims=True)
        
        # ----- Train Generator -----
        # Generator wants fake data to be classified as 1
        noise = np.random.randn(len(real_data), 2)
        fake_data = sigmoid(noise @ Wg + bg)
        fake_score = sigmoid(fake_data @ Wd + bd)
        g_loss = binary_cross_entropy(np.ones_like(fake_score), fake_score)
        
        # Simple gradient update for generator
        grad_g = noise.T @ ((fake_score - 1) @ Wd.T) * (fake_data * (1 - fake_data))
        Wg -= lr * grad_g / len(real_data)
        bg -= lr * np.mean((fake_score - 1) @ Wd.T * (fake_data * (1 - fake_data)), axis=0, keepdims=True)
        
        if epoch % 2 == 0:
            print(f"Epoch {epoch:2d} | D Loss: {d_loss:.4f} | G Loss: {g_loss:.4f}")
    
    # Generate new pattern
    test_noise = np.random.randn(1, 2)
    generated = sigmoid(test_noise @ Wg + bg)
    return np.round(generated).flatten()

# Compare both learning rates
print("="*50)
print("CORRECTED GAN IMPLEMENTATION")
print("="*50)

pattern_001 = train_gan_corrected(0.001)
pattern_01 = train_gan_corrected(0.01)

print("\n" + "="*50)
print("RESULTS")
print("="*50)
print(f"LR = 0.001 → Generated Pattern: {pattern_001}")
print(f"LR = 0.01  → Generated Pattern: {pattern_01}")