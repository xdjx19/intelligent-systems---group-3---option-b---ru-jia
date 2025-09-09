# MLP.py
# Train + evaluate a simple MLP on MNIST using CSV files in dataset/

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report

# ── 0) Reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ── 1) Load CSV data
train_data = np.loadtxt("dataset/mnist_train.csv", delimiter=",", dtype=np.float32)
test_data  = np.loadtxt("dataset/mnist_test.csv",  delimiter=",", dtype=np.float32)

# Split features and labels
Xtr = train_data[:, 1:] / 255.0  # normalize pixels to [0,1]
ytr = train_data[:, 0].astype(int)

Xte = test_data[:, 1:] / 255.0
yte = test_data[:, 0].astype(int)

# Convert to PyTorch tensors
Xtr = torch.tensor(Xtr, dtype=torch.float32)
ytr = torch.tensor(ytr, dtype=torch.long)
Xte = torch.tensor(Xte, dtype=torch.float32)
yte = torch.tensor(yte, dtype=torch.long)

# Data loaders
train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=64, shuffle=True)
test_loader  = DataLoader(TensorDataset(Xte, yte), batch_size=256, shuffle=False)

# ── 2) Define MLP
class SimpleMLP(nn.Module):
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.drop = nn.Dropout(0.25)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.drop(x)
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)
model = SimpleMLP().to(device)

# ── 3) Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ── 4) Helper functions
@torch.no_grad()
def eval_accuracy(loader: DataLoader) -> float:
    model.eval()
    correct = total = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb).argmax(1)
        correct += (pred == yb).sum().item()
        total += yb.size(0)
    return correct / total

def train_epochs(epochs: int = 10):
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

# ── 5) Main
if __name__ == "__main__":
    train_epochs(epochs=10)  # increase if you want higher accuracy

    # Final test accuracy
    final_acc = eval_accuracy(test_loader)
    print(f"\nFinal test accuracy: {final_acc:.4f}")

    # Confusion matrix and classification report
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

    print(f"\nVerified test accuracy over ALL {len(all_truth)} samples: {(all_preds==all_truth).mean():.4f}")
    print(f"Correct: {(all_preds==all_truth).sum()} / {len(all_truth)}")
    print("\nConfusion matrix:\n", confusion_matrix(all_truth, all_preds))
    print("\nClassification report:\n", classification_report(all_truth, all_preds))