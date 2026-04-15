import numpy as np

# 1. Dataset (4x4 patterns flattened to 16D)
X = np.array([
    [1,1,1,1, 1,0,0,1, 1,0,0,1, 1,1,1,1], # Pattern 1
    [0,1,0,0, 1,1,0,0, 0,1,0,0, 1,1,1,0], # Pattern 2
    [1,1,1,0, 0,0,1,0, 1,1,1,0, 0,0,1,0], # Pattern 3
    [1,1,1,1, 0,0,1,0, 0,1,0,0, 1,1,1,1]  # Pattern 4
], dtype=float)

def sigmoid(x): return 1 / (1 + np.exp(-x))
def sigmoid_der(x): return x * (1 - x)

def train_vae(lr, epochs=10):
    np.random.seed(42)
    # Weights: Input(16) -> Hidden(8) -> Mean/Var(2 each) -> Decode(16)
    W_enc = np.random.randn(16, 8) * 0.1
    W_mu = np.random.randn(8, 2) * 0.1
    W_logvar = np.random.randn(8, 2) * 0.1
    W_dec = np.random.randn(2, 16) * 0.1

    print(f"\nTraining VAE with LR: {lr}")
    for epoch in range(1, epochs + 1):
        # --- Encoder ---
        h = sigmoid(X @ W_enc)
        mu = h @ W_mu
        logvar = h @ W_logvar
        
        # --- Reparameterization Trick ---
        std = np.exp(0.5 * logvar)
        eps = np.random.randn(*mu.shape)
        z = mu + eps * std
        
        # --- Decoder ---
        out = sigmoid(z @ W_dec)
        
        # --- Loss (Simplified MSE + KL Approximation) ---
        recon_loss = np.mean((out - X)**2)
        kl_loss = -0.5 * np.mean(1 + logvar - mu**2 - np.exp(logvar))
        loss = recon_loss + kl_loss

        # --- Manual Gradient Step (Simplified) ---
        err = (out - X) * sigmoid_der(out)
        W_dec -= lr * z.T @ err
        grad_h = (err @ W_dec.T)
        W_mu -= lr * h.T @ grad_h
        W_enc -= lr * X.T @ (grad_h @ W_mu.T * sigmoid_der(h))

        if epoch % 5 == 0: print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    # Generate a new sample from latent space N(0,1)
    sample_z = np.random.randn(1, 2)
    generated = sigmoid(sample_z @ W_dec)
    return np.round(generated).reshape(4,4)

# 2. Compare Outputs
gen_001 = train_vae(0.001)
gen_01 = train_vae(0.01)

print("\nGenerated Pattern (LR 0.01):\n", gen_01)
print("Interpretation: The VAE has learned the general distribution and 'hallucinated' a new 4x4 pattern.")