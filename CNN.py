# CNN.py
# Train + evaluate a small CNN on MNIST using arrays prepared in Reading.py

import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report

# ── 0) Reproducibility (optional)
torch.manual_seed(42)
np.random.seed(42)

# ── 1) Get normalized data from Reading.py (keep Reading.py in same folder)
# Reading.py must define: train_norm, test_norm (float32 in [0,1]) and labels
from Reading import train_norm, test_norm, train_labels, test_labels  # noqa: E402

# ── 2) Build tensors / loaders for the WHOLE dataset
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

Xtr = torch.tensor(train_norm, dtype=torch.float32).unsqueeze(1)  # [60000,1,28,28]
ytr = torch.tensor(train_labels, dtype=torch.long)
Xte = torch.tensor(test_norm,  dtype=torch.float32).unsqueeze(1)  # [10000,1,28,28]
yte = torch.tensor(test_labels, dtype=torch.long)

train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=64, shuffle=True)
test_loader  = DataLoader(TensorDataset(Xte, yte), batch_size=256, shuffle=False)

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

model = SmallCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)  # add weight_decay=1e-4 if needed

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
    return correct / total

def train_epochs(epochs: int = 6):
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
        te_acc = eval_accuracy(test_loader)
        print(f"Epoch {ep:02d} | loss={running_loss/len(train_loader.dataset):.4f} "
              f"| train_acc={tr_acc:.4f} | test_acc={te_acc:.4f}")

if __name__ == "__main__":
    # ── 5) Train
    train_epochs(epochs=6)  # increase to 8–10 if you want ~99%+

    # ── 6) Final accuracy on ALL 10,000 test samples
    final_acc = eval_accuracy(test_loader)
    print(f"\nFinal test accuracy: {final_acc:.4f}")

    # ── 7) Save model
    save_path = "cnn_mnist.pt"
    torch.save(model.state_dict(), save_path)
    print(f"Saved model -> {save_path}")

    # ── 8) Full evaluation report (confusion matrix + per-class metrics)
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

    # ── 9) (Optional) Show a grid of misclassified samples for the report
    SHOW_MISCLASSIFIED = False  # set True if you want a window popup
    if SHOW_MISCLASSIFIED:
        import matplotlib.pyplot as plt
        mis_idx = np.where(all_preds != all_truth)[0][:32]
        if len(mis_idx) > 0:
            cols = 8
            rows = math.ceil(len(mis_idx) / cols)
            plt.figure(figsize=(cols*1.2, rows*1.2))
            for i, idx in enumerate(mis_idx):
                plt.subplot(rows, cols, i+1)
                plt.imshow(test_norm[idx], cmap="gray", vmin=0, vmax=1)
                plt.title(f"T:{all_truth[idx]} P:{all_preds[idx]}", fontsize=8)
                plt.axis("off")
            plt.suptitle("Misclassified test samples")
            plt.tight_layout()
            plt.show()
