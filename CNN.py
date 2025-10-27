# CNN_train_only.py
# Train a small CNN on MNIST train.csv without evaluating.

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# ── 0) CLI args
parser = argparse.ArgumentParser(description="Train CNN on MNIST train.csv only")
parser.add_argument("--train", type=str, default="mnist_train.csv", help="Path to train CSV")
parser.add_argument("--epochs", type=int, default=8, help="Training epochs")
parser.add_argument("--batch", type=int, default=64, help="Train batch size")
parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--save", type=str, default="cnn_mnist_trained.pt", help="File to save model weights")
args = parser.parse_args()

# ── 1) Reproducibility
# Set random seeds for libraries for improved results
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# ── 2) Load training CSV
def load_mnist_csv(path: str):
    assert os.path.exists(path), f"CSV not found: {path}"
    df = pd.read_csv(path)
    label_col = df.columns[0]  # assume first column is label
    y = df[label_col].to_numpy(dtype=np.int64) # Extract labels
    X = df.drop(columns=[label_col]).to_numpy(dtype=np.float32) / 255.0 # Normalize
    X = X.reshape(-1, 1, 28, 28) # Reshape to fit model input
    return X, y

# Load training data
Xtr_np, ytr_np = load_mnist_csv(args.train)
print(f"Loaded train set: {len(ytr_np)} rows from {args.train}")
print(f"Xtr shape: {Xtr_np.shape}")

# ── 3) Build tensors / loader
# Set the device for model training
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# Convert NumPy arrays to PyTorch tensors and create a DataLoader for batching
Xtr = torch.tensor(Xtr_np, dtype=torch.float32)
ytr = torch.tensor(ytr_np, dtype=torch.long)
train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=args.batch, shuffle=True)

# Calculate the number of classes for the output layer
num_classes = int(ytr_np.max()) + 1

# ── 4) CNN model
class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        # Initialize layers: two convolutional, pooling, dropout, and two fully connected
        self.c1 = nn.Conv2d(1, 32, 3, padding=1)
        self.c2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.drop = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Define forward pass
        x = self.pool(torch.relu(self.c1(x)))
        x = self.pool(torch.relu(self.c2(x)))
        x = self.drop(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Initialize the CNN model and set up the loss function and optimizer
model = SmallCNN(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

# ── 5) Train loop (only training, no evaluation)
def train_epochs(epochs: int):
    for ep in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)
            running_correct += (logits.argmax(1) == yb).sum().item()
            running_total += yb.size(0)

        tr_loss = running_loss / running_total
        tr_acc  = running_correct / running_total
        print(f"Epoch {ep:02d} | loss={tr_loss:.4f} | train_acc={tr_acc:.4f}")

# ── 6) Run training
train_epochs(args.epochs)

# ── 7) Save model weights
torch.save(model.state_dict(), args.save)
print(f"\nSaved trained model -> {args.save}")
