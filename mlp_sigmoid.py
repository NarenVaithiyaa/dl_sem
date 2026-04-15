import numpy as np
import matplotlib.pyplot as plt

# ============================================
# 1. DATASET
# ============================================
# Study Hours, Attendance
X = np.array([[2, 50],
              [4, 60],
              [6, 70],
              [8, 80]])

# Result: 0 = Fail, 1 = Pass
y = np.array([[0], [0], [1], [1]])

# Normalize data (important for sigmoid)
X[:, 0] = X[:, 0] / 10  # Study hours / 10
X[:, 1] = X[:, 1] / 100 # Attendance / 100

print("Normalized Data:")
print(X)
print("\nTargets:", y.flatten())

# ============================================
# 2. INITIALIZE WEIGHTS & BIASES
# ============================================
np.random.seed(42)  # For reproducible results

# Input (2) → Hidden (3)
W1 = np.random.randn(2, 3) * 0.5   # 2x3 matrix
b1 = np.zeros((1, 3))               # 1x3 bias

# Hidden (3) → Output (1)
W2 = np.random.randn(3, 1) * 0.5   # 3x1 matrix
b2 = np.zeros((1, 1))               # 1x1 bias

# ============================================
# 3. SIGMOID FUNCTIONS
# ============================================
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# ============================================
# 4. TRAINING PARAMETERS
# ============================================
learning_rate = 0.5
epochs = 50
losses = []

# ============================================
# 5. TRAINING LOOP (Forward + Backward)
# ============================================
for epoch in range(epochs):
    # ---------- FORWARD PROPAGATION ----------
    # Layer 1 (Input → Hidden)
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)
    
    # Layer 2 (Hidden → Output)
    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)  # Final prediction
    
    # Calculate loss (Mean Squared Error)
    loss = np.mean((y - a2) ** 2)
    losses.append(loss)
    
    # ---------- BACKPROPAGATION ----------
    # Output layer error
    d_z2 = (a2 - y) * sigmoid_derivative(a2)
    
    # Hidden layer error
    d_W2 = np.dot(a1.T, d_z2)
    d_b2 = np.sum(d_z2, axis=0, keepdims=True)
    
    d_a1 = np.dot(d_z2, W2.T)
    d_z1 = d_a1 * sigmoid_derivative(a1)
    
    d_W1 = np.dot(X.T, d_z1)
    d_b1 = np.sum(d_z1, axis=0, keepdims=True)
    
    # ---------- UPDATE WEIGHTS ----------
    W2 -= learning_rate * d_W2
    b2 -= learning_rate * d_b2
    W1 -= learning_rate * d_W1
    b1 -= learning_rate * d_b1
    
    # Print progress every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:2d}, Loss: {loss:.6f}")

# ============================================
# 6. EVALUATE ACCURACY
# ============================================
print("\n" + "="*50)
print("FINAL RESULTS")
print("="*50)

# Get final predictions
z1_final = np.dot(X, W1) + b1
a1_final = sigmoid(z1_final)
z2_final = np.dot(a1_final, W2) + b2
predictions = sigmoid(z2_final)

# Convert to 0 or 1 (threshold = 0.5)
predicted_class = (predictions >= 0.5).astype(int)

print("\nActual vs Predicted:")
print("Actual  Predicted  Probability")
print("-" * 35)
for i in range(len(y)):
    print(f"  {y[i][0]}        {predicted_class[i][0]}          {predictions[i][0]:.4f}")

# Calculate accuracy
accuracy = np.mean(predicted_class == y) * 100
print(f"\n Accuracy: {accuracy:.2f}%")

# ============================================
# 7. PREDICT FOR NEW STUDENT
# ============================================
print("\n" + "="*50)
print("PREDICTION FOR NEW STUDENT")
print("="*50)

# Study Hours = 5, Attendance = 65
new_student = np.array([[5, 65]])

# Normalize the same way
new_student[0, 0] = new_student[0, 0] / 10  # Hours
new_student[0, 1] = new_student[0, 1] / 100 # Attendance

# Forward pass
z1_new = np.dot(new_student, W1) + b1
a1_new = sigmoid(z1_new)
z2_new = np.dot(a1_new, W2) + b2
prediction = sigmoid(z2_new)

print(f"\nStudent: Study Hours = 5, Attendance = 65%")
print(f"Probability of passing: {prediction[0][0]:.4f} ({prediction[0][0]*100:.1f}%)")

if prediction >= 0.5:
    print(" PREDICTION: PASS")
else:
    print(" PREDICTION: FAIL")

# ============================================
# 8. PLOT LOSS CURVE
# ============================================
plt.figure(figsize=(10, 4))
plt.plot(losses)
plt.title('Loss Reduction over Epochs')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.grid(True)
plt.show()
