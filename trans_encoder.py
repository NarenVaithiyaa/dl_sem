import torch
import torch.nn as nn
import numpy as np

# 1. Dataset (4 samples, 3 words/seq_len, 4 dims)
X = torch.tensor([
    [[1, 0, 1, 0], [1, 1, 1, 0], [1, 0, 1, 1]], # Pos 1
    [[0, 1, 1, 1], [1, 1, 0, 1], [1, 1, 1, 1]], # Pos 2
    [[0, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0]], # Neg 1
    [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0]]  # Neg 2
], dtype=torch.float32)
y = torch.tensor([[1], [1], [0], [0]], dtype=torch.float32)

# 2. Transformer Model
class TransformerClassifier(nn.Module):
    def __init__(self, d_model=4, nhead=2, dim_feedforward=8):
        super().__init__()
        self.pos_encoding = nn.Parameter(torch.randn(1, 3, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = x + self.pos_encoding
        x = self.transformer(x)
        x = x.mean(dim=1) # Global Average Pooling
        return torch.sigmoid(self.fc(x))

def train_transformer(lr, epochs=10):
    model = TransformerClassifier()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    print(f"\nTraining Transformer with LR: {lr}")
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        if epoch % 5 == 0: print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    preds = (outputs > 0.5).float()
    acc = (preds == y).float().mean()
    return model, acc.item()

# 3. Execution & Comparison
m1, acc1 = train_transformer(0.001)
m2, acc2 = train_transformer(0.01)

# 4. Predict for New Review
new_review = torch.tensor([[[0, 1, 0, 1], [1, 1, 1, 1], [0, 1, 1, 0]]], dtype=torch.float32)
with torch.no_grad():
    pred = m2(new_review)
    sentiment = "Positive" if pred > 0.5 else "Negative"

print(f"\nAccuracy (LR 0.01): {acc2*100:.2f}%")
print(f"New Review Prediction: {pred.item():.4f} ({sentiment})")