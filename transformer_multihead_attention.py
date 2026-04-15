import numpy as np

# 1. Dataset (Toy sequences)
X = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]], dtype=float)
seq_len, d_model = X.shape

# 2. Positional Encoding
def get_positional_encoding(seq_len, d_model):
    pos_enc = np.zeros((seq_len, d_model))
    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            pos_enc[pos, i] = np.sin(pos / (10000 ** (i / d_model)))
            if i + 1 < d_model:
                pos_enc[pos, i + 1] = np.cos(pos / (10000 ** (i / d_model)))
    return pos_enc

# 3. Scaled Dot-Product Attention
def scaled_dot_product_attention(Q, K, V):
    d_k = Q.shape[-1]
    scores = np.dot(Q, K.T) / np.sqrt(d_k)
    weights = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)
    return np.dot(weights, V), weights

# 4. Multi-Head Attention (Simplified)
class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = np.random.randn(d_model, d_model) * 0.5
        self.W_k = np.random.randn(d_model, d_model) * 0.5
        self.W_v = np.random.randn(d_model, d_model) * 0.5
        self.W_o = np.random.randn(d_model, d_model) * 0.5

    def forward(self, x):
        # In a real model, we would reshape and transpose for heads.
        # Simplified version: Treat heads as chunks of the dimension.
        Q = np.dot(x, self.W_q)
        K = np.dot(x, self.W_k)
        V = np.dot(x, self.W_v)
        
        attn_out, weights = scaled_dot_product_attention(Q, K, V)
        return np.dot(attn_out, self.W_o), weights

# 5. Transformer Encoder Block
class TransformerEncoderBlock:
    def __init__(self, d_model, num_heads):
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.W_out = np.random.randn(d_model, d_model) * 0.5 # For next-vector prediction

    def forward(self, x):
        attn_output, weights = self.mha.forward(x)
        out = attn_output + x  # Residual connection
        prediction = np.dot(out, self.W_out)
        return prediction, weights

# Execution
pos_encoding = get_positional_encoding(seq_len, d_model)
X_input = X + pos_encoding

model = TransformerEncoderBlock(d_model, num_heads=1) # Simplified to 1 head for d_model=3
prediction, attn_weights = model.forward(X_input)

print("Attention Weights:")
print(attn_weights)
print("\nPredicted Next Vector Sequence:")
print(prediction)
print("\nFinal Predicted Vector:")
print(np.round(prediction[-1], 2))