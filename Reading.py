# Reading.py
# Load MNIST IDX files, build preprocessed arrays, and (optionally) visualize.
# - Safe to import from other files (no plots on import)
# - Run directly to see sample images

import struct
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

# -----------------------
# 1) IDX Reader Functions
# -----------------------
def load_images(path: str) -> np.ndarray:
    """Read IDX3 images -> (N, 28, 28) uint8."""
    with open(path, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(num, rows, cols)

def load_labels(path: str) -> np.ndarray:
    """Read IDX1 labels -> (N,) uint8."""
    with open(path, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels

# -----------------------
# 2) Preprocessing helpers
# -----------------------
def normalize01(x_uint8: np.ndarray) -> np.ndarray:
    """Scale uint8 [0..255] -> float32 [0..1]."""
    return x_uint8.astype(np.float32) / 255.0

def binarize_otsu(batch_uint8: np.ndarray) -> np.ndarray:
    """Otsu threshold per image -> uint8 {0,255}."""
    out = np.empty_like(batch_uint8)
    for i in range(batch_uint8.shape[0]):
        _, th = cv2.threshold(batch_uint8[i], 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        out[i] = th
    return out

def denoise_open(batch_uint8_bin: np.ndarray, ksize: int = 2) -> np.ndarray:
    """Morphological opening to remove tiny noise (expects binary images)."""
    kernel = np.ones((ksize, ksize), np.uint8)
    out = np.empty_like(batch_uint8_bin)
    for i in range(batch_uint8_bin.shape[0]):
        out[i] = cv2.morphologyEx(batch_uint8_bin[i], cv2.MORPH_OPEN, kernel)
    return out

def show_row(title: str, arr: np.ndarray, labels: np.ndarray, cmap: str = "gray") -> None:
    """Visualize first 10 images with labels."""
    plt.figure(figsize=(10, 2))
    for i in range(10):
        ax = plt.subplot(1, 10, i + 1)
        # keep consistent scaling
        if arr.dtype == np.float32:
            ax.imshow(arr[i], cmap=cmap, vmin=0, vmax=1)
        else:
            ax.imshow(arr[i], cmap=cmap, vmin=0, vmax=255)
        ax.set_title(int(labels[i]))
        ax.axis("off")
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

# -----------------------
# 3) Point to your files
# -----------------------
# <<< EDIT THIS PATH to your folder with the four IDX files >>>
data_dir = r"C:\Users\arsal\OneDrive\Desktop\Documents\IS AST"

# Build absolute paths
train_images_path = os.path.join(data_dir, "train-images.idx3-ubyte")
train_labels_path = os.path.join(data_dir, "train-labels.idx1-ubyte")
test_images_path  = os.path.join(data_dir, "t10k-images.idx3-ubyte")
test_labels_path  = os.path.join(data_dir, "t10k-labels.idx1-ubyte")

# -----------------------
# 4) Load + preprocess (module-level variables for import)
# -----------------------
# Raw data (uint8)
train_images: np.ndarray = load_images(train_images_path)
train_labels: np.ndarray = load_labels(train_labels_path)
test_images:  np.ndarray = load_images(test_images_path)
test_labels:  np.ndarray = load_labels(test_labels_path)

# Normalized (float32)
train_norm: np.ndarray = normalize01(train_images)
test_norm:  np.ndarray = normalize01(test_images)

# Optional: binary & denoised versions (uint8)
train_bin: np.ndarray = binarize_otsu(train_images)
test_bin:  np.ndarray = binarize_otsu(test_images)
train_bin_clean: np.ndarray = denoise_open(train_bin, ksize=2)
test_bin_clean:  np.ndarray = denoise_open(test_bin,  ksize=2)

# -----------------------
# 5) Only visualize when run directly (not on import)
# -----------------------
if __name__ == "__main__":
    print("Train set:", train_images.shape, train_labels.shape)
    print("Test set :", test_images.shape,  test_labels.shape)
    print("Before:", train_images.dtype, train_images.min(), train_images.max())
    print("After :", train_norm.dtype,  train_norm.min(),  train_norm.max())

    show_row("Original", train_images, train_labels)
    show_row("Normalized (0..1)", train_norm, train_labels)
    show_row("Binarized (Otsu)", train_bin, train_labels)
    show_row("Denoised binary", train_bin_clean, train_labels)
