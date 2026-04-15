import numpy as np
from torchvision import datasets, transforms

# 1. Load a small sample of MNIST
mist_data = datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor())
X_train = mist_data.data[:10].numpy() / 255.0  # 10 images for speed
y_train = mist_data.targets[:10].numpy()

def pool2d(X, size=2):
    h, w = X.shape
    res = np.zeros((h // size, w // size))
    for i in range(0, h // size):
        for j in range(0, w // size):
            res[i, j] = np.max(X[i*size:i*size+size, j*size:j*size+size])
    return res

def convolve2d(X, kernel):
    xh, xw = X.shape
    kh, kw = kernel.shape
    res = np.zeros((xh - kh + 1, xw - kw + 1))
    for i in range(res.shape[0]):
        for j in range(res.shape[1]):
            res[i, j] = np.sum(X[i:i+kh, j:j+kw] * kernel)
    return res

# 2. Simple 'Scratch' Forward Pass
np.random.seed(42)
kernel = np.random.randn(3, 3) * 0.1
weights = np.random.randn(169, 10) * 0.1 # After 3x3 conv and 2x2 pool

print("Processing batch...")
correct = 0
for img, label in zip(X_train, y_train):
    # Conv (28x28 -> 26x26)
    feat_map = convolve2d(img, kernel)
    # ReLU
    feat_map = np.maximum(0, feat_map)
    # Pool (26x26 -> 13x13)
    pooled = pool2d(feat_map, size=2)
    # Flatten and FC
    out = pooled.flatten() @ weights
    pred = np.argmax(out)
    if pred == label: correct += 1

print(f"Untrained 'From Scratch' Accuracy on 10 samples: {correct/10 * 100}%")