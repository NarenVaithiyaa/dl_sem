import numpy as np

# 1. Dataset
X = np.array([[1, 0, 1, 0, 1]], dtype=float).T  # (5, 1)

def compute_attention(lr, steps=10):
    np.random.seed(42)
    # Initialize Q, K, V weights
    Wq, Wk, Wv = [np.random.randn(1, 1) * 0.5 for _ in range(3)]
    
    for step in range(1, steps + 1):
        # Step 1: Compute Q, K, V
        Q, K, V = X @ Wq, X @ Wk, X @ Wv
        
        # Step 2: Attention Scores (Q @ K.T)
        scores = (Q @ K.T) / np.sqrt(1)
        
        # Step 3: Softmax to get Weights
        exp_scores = np.exp(scores - np.max(scores))
        weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        
        # Step 4: Context Vector
        context = weights @ V
        
        # Step 5: Manual Update (Simplified gradient step towards the signal)
        error = X - context
        Wq += lr * np.sum(error)
        Wv += lr * np.sum(error)
    print("Query : ",Q)
    print("\nKey : ",K)
    print("\nValue : ",V)
                
    return weights, context

# 2. Execute for different Learning Rates
weights_001, _ = compute_attention(0.001)
weights_01, _ = compute_attention(0.01)

# 3. Identify most important pixel (from the last step's weights)
importance = np.mean(weights_01, axis=0)
most_important_idx = np.argmax(importance)


print("Attention Weights (LR 0.01):\n", np.round(weights_01, 3))
print(f"\nAverage Importance per Pixel: {np.round(importance, 3)}")
print(f"The most important pixel index is: {most_important_idx} (Value: {X[most_important_idx][0]})")