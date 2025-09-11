# CNN_from_CSV.py
# Train + evaluate a small CNN on MNIST directly from a CSV file (e.g., Kaggle format)
# Usage:
#   python CNN_from_CSV.py --csv mnist_train.csv --epochs 8 --batch 64

import os
import math
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# ── 0) CLI args
parser = argparse.ArgumentParser(description="Train a CNN on MNIST from CSV")
parser.add_argument("--csv", type=str, default="mnist_train.csv", help="Path to MNIST CSV (label + 784 pixels)")
parser.add_argument("--epochs", type=int, default=8, help="Training epochs")
parser.add_argument("--batch", type=int, default=64, help="Train batch size")
parser.add_argument("--test-batch", type=int, default=256, help="Eval batch size")
parser.add_argument("--val-split", type=float, default=0.1, help="Holdout fraction for testing (0.0–0.5)")
parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--save", type=str, default="cnn_mnist_from_csv.pt", help="Where to save model weights")
args = parser.parse_args()

# ── 1) Reproducibility
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# ── 2) Read CSV -> tensors
# Expected: first column = label, next 784 columns = pixel intensities [0..255]
assert os.path.exists(args.csv), f"CSV not found: {args.csv}"
df = pd.read_csv(args.csv)

# Try to be robust if the CSV has named columns like 'label','pixel0'...'pixel783'
# Infer label column name (default to first column)
label_col = df.columns[0]
y_np = df[label_col].to_numpy(dtype=np.int64)

# Remaining columns are pixels
X_cols = [c for c in df.columns if c != label_col]
X_np = df[X_cols].to_numpy(dtype=np.float32)

# Normalize to [0,1] and reshape to N x 1 x 28 x 28
X_np = (X_np / 255.0).reshape(-1, 1, 28, 28)

print(f"Loaded {len(df)} rows from {args.csv}")
print(f"X shape: {X_np.shape}  y shape: {y_np.shape}  labels in [0..{int(y_np.max())}]")

# Split train/test
test_size = args.val_split
Xtr_np, Xte_np, ytr_np, yte_np = train_test_split(
    X_np, y_np, test_size=test_size, random_state=args.seed, stratify=y_np
) if test_size > 0 else (X_np, X_np[:0], y_np, y_np[:0])

# Build tensors / loaders
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

Xtr = torch.tensor(Xtr_np, dtype=torch.float32)
ytr = torch.tensor(ytr_np, dtype=torch.long)
Xte = torch.tensor(Xte_np, dtype=torch.float32)
yte = torch.tensor(yte_np, dtype=torch.long)

train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=args.batch, shuffle=True, drop_last=False)
test_loader  = DataLoader(TensorDataset(Xte, yte), batch_size=args.test_batch, shuffle=False, drop_last=False) if len(yte_np) else None

# ── 3) Define a compact CNN (LeNet-ish)
class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.c1 = nn.Conv2d(1, 32, 3, padding=1)     # -> 32x28x28
        self.c2 = nn.Conv2d(32, 64, 3, padding=1)    # -> 64x14x14 after pool
        self.pool = nn.MaxPool2d(2, 2)
        self.drop = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.c1(x)))        # 32x14x14
        x = self.pool(torch.relu(self.c2(x)))        # 64x7x7
        x = self.drop(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

model = SmallCNN(num_classes=int(y_np.max()) + 1).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)  # add weight_decay=1e-4 if needed

# ── 4) Training / evaluation helpers
@torch.no_grad()
def eval_accuracy(loader: DataLoader) -> float:
    model.eval()
    correct = total = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb).argmax(1)
        correct += (pred == yb).sum().item()
        total   += yb.size(0)
    return correct / total if total else float("nan")

def train_epochs(epochs: int):
    for ep in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)

        tr_acc = eval_accuracy(train_loader)
        te_acc = eval_accuracy(test_loader) if test_loader is not None else float("nan")
        print(f"Epoch {ep:02d} | loss={running_loss/len(train_loader.dataset):.4f} "
              f"| train_acc={tr_acc:.4f} | test_acc={te_acc:.4f}")

# ── 5) Train
train_epochs(args.epochs)

# ── 6) Final evaluation
if test_loader is not None:
    final_acc = eval_accuracy(test_loader)
    print(f"\nFinal test accuracy: {final_acc:.4f}")

# ── 7) Save model
torch.save(model.state_dict(), args.save)
print(f"Saved model -> {args.save}")

# ── 8) Full evaluation report (if we have a test split)
if test_loader is not None:
    model.eval()
    all_preds, all_truth = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            pred = model(xb).argmax(1).cpu().numpy()
            all_preds.append(pred)
            all_truth.append(yb.numpy())
    all_preds = np.concatenate(all_preds)
    all_truth = np.concatenate(all_truth)

    acc = (all_preds == all_truth).mean()
    print(f"\nVerified test accuracy over ALL {len(all_truth)} samples: {acc:.4f}")
    print(f"Correct: {(all_preds == all_truth).sum()} / {len(all_truth)}")

    cm = confusion_matrix(all_truth, all_preds)
    print("\nConfusion matrix (rows=true, cols=pred):\n", cm)
    print("\nPer-class precision/recall/F1:\n", classification_report(all_truth, all_preds))

# ── 9) (Optional) Toggle to visualize misclassified samples (requires matplotlib)
SHOW_MISCLASSIFIED = False
if SHOW_MISCLASSIFIED and test_loader is not None:
    import matplotlib.pyplot as plt
    mis_idx = np.where(all_preds != all_truth)[0][:32]
    if len(mis_idx) > 0:
        cols = 8
        rows = math.ceil(len(mis_idx) / cols)
        # We need access to the raw normalized test images:
        Xte_img = Xte.cpu().numpy().squeeze(1)  # [N,28,28]
        plt.figure(figsize=(cols*1.2, rows*1.2))
        for i, idx in enumerate(mis_idx):
            plt.subplot(rows, cols, i+1)
            plt.imshow(Xte_img[idx], cmap="gray", vmin=0, vmax=1)
            plt.title(f"T:{all_truth[idx]} P:{all_preds[idx]}", fontsize=8)
            plt.axis("off")
        plt.suptitle("Misclassified test samples")
        plt.tight_layout()
        plt.show()
