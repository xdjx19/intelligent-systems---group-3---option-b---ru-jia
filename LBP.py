"""
LBP.py
Local Binary Patterns (LBP) + Linear SVM baseline on MNIST.

Primary source: Reading.py (loads IDX files).
Fallback: dataset/*.csv if IDX files are unavailable (splits test CSV if needed).

"""

import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import os

def _load_from_reading():
    """Try loading arrays from Reading.py. Returns tuple or None on failure."""
    try:
        from Reading import (
            train_norm as _train_norm,
            test_norm as _test_norm,
            train_labels as _train_labels,
            test_labels as _test_labels,
        )  # noqa: E402
        # Basic shape checks
        assert _train_norm.ndim == 3 and _train_norm.shape[1:] == (28, 28)
        assert _test_norm.ndim == 3 and _test_norm.shape[1:] == (28, 28)
        return _train_norm, _test_norm, _train_labels, _test_labels
    except Exception as e:
        print("Reading.py unavailable or IDX files missing; falling back to CSV.")
        print(f"Reason: {e}")
        return None


def _load_from_csv():
    """Load MNIST from CSV files in dataset/. If train CSV is missing,
    split the test CSV into train/test.
    Returns: (train_imgs[Ntr,28,28], test_imgs[Nte,28,28], train_y[Ntr], test_y[Nte])
    """
    test_csv = os.path.join("dataset", "mnist_test.csv")
    train_csv = os.path.join("dataset", "mnist_train.csv")

    if not os.path.isfile(test_csv):
        raise FileNotFoundError("dataset/mnist_test.csv not found; cannot fallback.")

    # Helper to load a CSV to (imgs, labels)
    def read_csv(path):
        arr = np.loadtxt(path, delimiter=",", dtype=np.float32)
        y = arr[:, 0].astype(np.int64)
        X = arr[:, 1:] / 255.0
        imgs = X.reshape(-1, 28, 28)
        return imgs, y

    # Determine if train CSV is the real dataset (not a small link file)
    has_train = os.path.isfile(train_csv) and os.path.getsize(train_csv) > 1_000_000

    test_imgs, test_y = read_csv(test_csv)
    if has_train:
        train_imgs, train_y = read_csv(train_csv)
    else:
        # Split test into train/test (80/20 stratified)
        Xtr, Xte, ytr, yte = train_test_split(
            test_imgs, test_y, test_size=0.2, random_state=42, stratify=test_y
        )
        train_imgs, train_y = Xtr, ytr
        test_imgs, test_y = Xte, yte
        print("Note: Using a train/test split from mnist_test.csv (no train CSV).")

    return train_imgs, test_imgs, train_y, test_y


def lbp_hist(img: np.ndarray) -> np.ndarray:
    """Compute basic 3x3 LBP histogram (256 bins) for a single 2D image.

    - img: 2D array (float32 in [0,1] or uint8)
    - returns: (256,) float32 normalized histogram
    """
    # Ensure float32
    img = img.astype(np.float32)

    # Pad by 1 so border comparisons work
    p = np.pad(img, 1, mode="edge")

    c = p[1:-1, 1:-1]
    # 8 neighbors (clockwise starting top-left)
    n0 = p[0:-2, 0:-2]
    n1 = p[0:-2, 1:-1]
    n2 = p[0:-2, 2:  ]
    n3 = p[1:-1, 2:  ]
    n4 = p[2:  , 2:  ]
    n5 = p[2:  , 1:-1]
    n6 = p[2:  , 0:-2]
    n7 = p[1:-1, 0:-2]

    # Compare neighbors to center (>= to be stable on binaries)
    b0 = (n0 >= c).astype(np.uint8)
    b1 = (n1 >= c).astype(np.uint8)
    b2 = (n2 >= c).astype(np.uint8)
    b3 = (n3 >= c).astype(np.uint8)
    b4 = (n4 >= c).astype(np.uint8)
    b5 = (n5 >= c).astype(np.uint8)
    b6 = (n6 >= c).astype(np.uint8)
    b7 = (n7 >= c).astype(np.uint8)

    # Bit pack into 0..255
    code = (
        (b0 << 7)
        | (b1 << 6)
        | (b2 << 5)
        | (b3 << 4)
        | (b4 << 3)
        | (b5 << 2)
        | (b6 << 1)
        | (b7 << 0)
    )

    # Histogram over codes
    hist = np.bincount(code.ravel(), minlength=256).astype(np.float32)
    # Normalize to unit sum to be intensity-invariant
    s = hist.sum()
    if s > 0:
        hist /= s
    return hist


def batch_lbp(X: np.ndarray) -> np.ndarray:
    """Compute LBP histograms for a batch of images (N, H, W) -> (N, 256)."""
    N = X.shape[0]
    feats = np.empty((N, 256), dtype=np.float32)
    for i in range(N):
        feats[i] = lbp_hist(X[i])
    return feats


def main():
    loaded = _load_from_reading()
    if loaded is not None:
        train_imgs, test_imgs, train_labels, test_labels = loaded
    else:
        train_imgs, test_imgs, train_labels, test_labels = _load_from_csv()

    # Features
    Xtr = batch_lbp(train_imgs)
    Xte = batch_lbp(test_imgs)

    # Train a linear SVM; C can be tuned (e.g., 0.1, 1.0)
    clf = LinearSVC(C=1.0, max_iter=5000)
    clf.fit(Xtr, train_labels)

    # Evaluate
    acc_tr = clf.score(Xtr, train_labels)
    acc_te = clf.score(Xte, test_labels)
    print(f"Train acc: {acc_tr:.4f} | Test acc: {acc_te:.4f}")

    preds = clf.predict(Xte)
    print("\nConfusion matrix (rows=true, cols=pred):\n", confusion_matrix(test_labels, preds))
    print("\nPer-class precision/recall/F1:\n", classification_report(test_labels, preds))


if __name__ == "__main__":
    main()
