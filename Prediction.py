# Prediction.py
# Load trained CNN model and run predictions on mnist_test.csv

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ── 0) CLI args
parser = argparse.ArgumentParser(description="Run predictions on MNIST test CSV using trained model")
parser.add_argument("--test", type=str, default=os.path.join("dataset", "mnist_test.csv"), help="Path to test CSV")
parser.add_argument("--weights", type=str, default="cnn_mnist_trained.pt", help="Path to trained model weights")  # ✅ fixed name
parser.add_argument("--out", type=str, default="predictions.csv", help="File to save predictions")
args = parser.parse_args()

# ── 1) Load test data
def load_mnist_csv(path: str):
    assert os.path.exists(path), f"CSV not found: {path}"
    df = pd.read_csv(path)
    label_col = df.columns[0]  # first column contains labels
    y = df[label_col].to_numpy(dtype=np.int64)
    X = df.drop(columns=[label_col]).to_numpy(dtype=np.float32) / 255.0 # Normalise pixel values
    X = X.reshape(-1, 1, 28, 28) # CNN input reshaping
    return X, y

X, y_true = load_mnist_csv(args.test)
print(f"Loaded test set: {len(y_true)} rows from {args.test}")
print(f"X shape: {X.shape}")

# ── 2) CNN model (same as training)
class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        # Define layers
        self.c1 = nn.Conv2d(1, 32, 3, padding=1)
        self.c2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.drop = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Forward pass
        x = self.pool(torch.relu(self.c1(x)))
        x = self.pool(torch.relu(self.c2(x)))
        x = self.drop(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# ── 3) Load model + weights
device = "cuda" if torch.cuda.is_available() else "cpu" # Set model device
model = SmallCNN().to(device)

assert os.path.exists(args.weights), f"❌ Model weights not found: {args.weights}"
model.load_state_dict(torch.load(args.weights, map_location=device))
model.eval()
print(f"✅ Loaded model weights from {args.weights}")

# ── 4) Run predictions
with torch.no_grad():
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device) # Convert input for tensor library
    outputs = model(X_tensor).argmax(1).cpu().numpy() # Get class predictions

# ── 5) Evaluate
acc = accuracy_score(y_true, outputs)
print(f"\n✅ Test Accuracy: {acc:.4f}")
print("\nConfusion Matrix:\n", confusion_matrix(y_true, outputs))
print("\nPer-class Report:\n", classification_report(y_true, outputs))

# ── 6) Save predictions
out_df = pd.DataFrame({"TrueLabel": y_true, "Predicted": outputs})
out_df.to_csv(args.out, index=False)
print(f"\n💾 Saved predictions -> {args.out}")