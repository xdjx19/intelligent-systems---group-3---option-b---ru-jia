# gui.py
# Tkinter GUI that runs:
#   1) threshold_segmentation.segment_digits -> boxes
#   2) center/pad each box to 28x28
#   3) PyTorch SmallCNN (same as in CNN_train_only.py) -> predictions
#
# pip install opencv-python pillow numpy torch tensorflow

import os
import sys
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

import numpy as np
import cv2
import torch
import torch.nn as nn
import tensorflow as tf

# -------------------- import your segmentation --------------------
try:
    import threshold_segmentation as seg # Attempt to import segmentation module
except Exception as e:
    raise RuntimeError(
        "Could not import threshold_segmentation.py. "
        "Place gui.py in the same folder and ensure the file name is correct."
    ) from e

from train_math_solver import predict_image_segments, load_dataset

# -------------------- CNN model (must match CNN_train_only.py) --------------------
class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        # Define layers: two convolutional layers, pooling, dropout, and fully connected layers
        self.c1 = nn.Conv2d(1, 32, 3, padding=1)      # -> 32x28x28
        self.c2 = nn.Conv2d(32, 64, 3, padding=1)     # -> 64x14x14 after pool
        self.pool = nn.MaxPool2d(2, 2)
        self.drop = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Define forward pass
        x = self.pool(torch.relu(self.c1(x)))         # 32x14x14
        x = self.pool(torch.relu(self.c2(x)))         # 64x7x7
        x = self.drop(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

def load_weights(path, device):
    # Load weights from path and set model for evaluation
    model = SmallCNN(num_classes=10).to(device)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model

# -------------------- preprocessing helpers --------------------
def to_28x28_centered(img_bgr_or_gray):
    """Binarize (Otsu INV), tight-crop, pad to square, resize to 28x28."""
    if img_bgr_or_gray.ndim == 3:
        gray = cv2.cvtColor(img_bgr_or_gray, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_bgr_or_gray

    _, th_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ys, xs = np.where(th_inv > 0)
    if len(xs) == 0:
        return cv2.resize(th_inv, (28, 28), interpolation=cv2.INTER_AREA) # Resize if no digits found

    # Crop the digit and pad it to a square shape
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    digit = th_inv[y1:y2+1, x1:x2+1]

    h, w = digit.shape
    s = max(h, w)
    pad_y = (s - h) // 2
    pad_x = (s - w) // 2
    square = np.zeros((s, s), dtype=np.uint8)
    square[pad_y:pad_y+h, pad_x:pad_x+w] = digit

    return cv2.resize(square, (28, 28), interpolation=cv2.INTER_AREA)

def draw_boxes(img_bgr, boxes, color=(0, 255, 0)):
    # Draw bounding boxes around individual digits
    vis = img_bgr.copy()
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
    return vis

# -------------------- GUI --------------------
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Digit Recognition (Threshold Segmentation + CNN)")
        self.root.geometry("1150x780")

        self.img_bgr = None
        self.image_path = None
        self.preview_bgr = None
        self.boxes = []
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.model_path = None

        # Top bar
        top = tk.Frame(root)
        top.pack(fill=tk.X, padx=10, pady=8)

        tk.Button(top, text="Open Image", command=self.open_image).pack(side=tk.LEFT, padx=4)
        self.img_label = tk.Label(top, text="No image loaded")
        self.img_label.pack(side=tk.LEFT, padx=6)

        tk.Button(top, text="Load CNN Weights (.pt)", command=self.load_model).pack(side=tk.LEFT, padx=14)
        self.model_label = tk.Label(top, text=f"Model: (none) | Device: {self.device}")
        self.model_label.pack(side=tk.LEFT, padx=6)

        tk.Button(top, text="Run", command=self.run_all).pack(side=tk.LEFT, padx=16)
        tk.Button(top, text="Math Solver", command=self.run_math_solver).pack(side=tk.LEFT, padx=4)
        tk.Button(top, text="Save Preview", command=self.save_preview).pack(side=tk.LEFT, padx=4)
        tk.Button(top, text="Save Raw Crops", command=self.save_crops_raw).pack(side=tk.LEFT, padx=4)
        tk.Button(top, text="Save 28x28 Crops", command=self.save_crops_28).pack(side=tk.LEFT, padx=4)

        # Status + predictions
        status = tk.Frame(root)
        status.pack(fill=tk.X, padx=10, pady=4)
        self.pred_var = tk.StringVar(value="Predictions: —")
        self.stat_var = tk.StringVar(value="Ready")
        tk.Label(status, textvariable=self.pred_var, font=("Segoe UI", 12, "bold")).pack(side=tk.LEFT)
        tk.Label(status, textvariable=self.stat_var, fg="#666").pack(side=tk.RIGHT)

        # Canvas
        self.canvas_w, self.canvas_h = 1080, 620
        self.canvas = tk.Canvas(root, bg="#111", width=self.canvas_w, height=self.canvas_h, highlightthickness=0)
        self.canvas.pack(padx=8, pady=8)
        self.tk_img = None

    # ---------- actions ----------
    def open_image(self):
        path = filedialog.askopenfilename(
            title="Open image",
            filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff")]
        )
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Error", f"Failed to read image:\n{path}")
            return
        self.image_path = path
        self.img_bgr = img
        self.preview_bgr = None
        self.boxes = []
        self.img_label.config(text=os.path.basename(path))
        self.pred_var.set("Predictions: —")
        self.stat_var.set("Image loaded")
        self._show(img)

    def load_model(self):
        path = filedialog.askopenfilename(
            title="Open model weights (.pt)",
            filetypes=[("PyTorch weights", "*.pt;*.pth;*.bin;*.*")]
        )
        if not path:
            return
        try:
            self.model = load_weights(path, self.device)
            self.model_path = path
            self.model_label.config(text=f"Model: {os.path.basename(path)} | Device: {self.device}")
            self.stat_var.set("Model loaded")
        except Exception as e:
            messagebox.showerror("Model load error", f"{e}\n\n{traceback.format_exc()}")

    def run_all(self):
        if self.img_bgr is None:
            messagebox.showwarning("Missing image", "Open an image first.")
            return
        if self.model is None:
            messagebox.showwarning("Missing model", "Load CNN weights first.")
            return

        self.stat_var.set("Running…")
        self.root.update_idletasks()

        try:
            # 1) Segment with your threshold_segmentation
            boxes, bin_img = seg.segment_digits(self.img_bgr)
            if hasattr(seg, "tighten_boxes"):
                boxes = seg.tighten_boxes(bin_img, boxes, margin=2)

            if not boxes:
                self.pred_var.set("Predictions: (no digits found)")
                self._show(self.img_bgr)
                self.stat_var.set("Done")
                return

            # 2) Draw preview
            vis = seg.draw_boxes(self.img_bgr, boxes) if hasattr(seg, "draw_boxes") else draw_boxes(self.img_bgr, boxes)
            self.preview_bgr = vis.copy()
            self.boxes = boxes
            self._show(vis)

            # 3) Build batch (N,1,28,28) and predict
            crops_28 = []
            for (x1, y1, x2, y2) in boxes:
                crop = self.img_bgr[y1:y2, x1:x2]
                d28 = to_28x28_centered(crop)
                crops_28.append(d28)

            X = np.stack(crops_28, axis=0).astype(np.float32) / 255.0
            X = np.expand_dims(X, axis=1)
            xt = torch.from_numpy(X).to(self.device)

            with torch.no_grad():
                logits = self.model(xt)
                preds = torch.argmax(logits, dim=1).cpu().numpy()

            pred_str = "".join(str(int(p)) for p in preds)
            self.pred_var.set(f"Predictions (left→right): {list(map(int,preds))} | as string: {pred_str}")
            self.stat_var.set("Done")

        except Exception as e:
            self.stat_var.set("Error")
            messagebox.showerror("Run error", f"{e}\n\n{traceback.format_exc()}")

    def run_math_solver(self):
        if self.img_bgr is None or self.image_path is None:
            messagebox.showwarning("Missing image", "Open an image first.")
            return

        model_path = "math_solver_model.keras"
        if not os.path.exists(model_path):
            messagebox.showerror("Model Missing", f"Math solver model not found at: {model_path}")
            return

        try:
            model = tf.keras.models.load_model(model_path)
            _, _, classes = load_dataset()  # ✅ FIXED unpacking
            result = predict_image_segments(self.image_path, model, classes)

            # Handle None or string outputs gracefully
            if result is None:
                result = "No valid expression detected."
            messagebox.showinfo("Math Solver Output", f"Result: {result}")
            self.stat_var.set("Math solver done")
        except Exception as e:
            self.stat_var.set("Math solver error")
            messagebox.showerror("Math Solver Error", f"{e}\n\n{traceback.format_exc()}")

    def save_preview(self):
        if self.preview_bgr is None:
            messagebox.showinfo("Info", "Nothing to save. Run first.")
            return
        path = filedialog.asksaveasfilename(
            title="Save preview",
            defaultextension=".png",
            filetypes=[("PNG", "*.png")]
        )
        if not path:
            return
        cv2.imwrite(path, self.preview_bgr)
        messagebox.showinfo("Saved", f"Preview saved to:\n{path}")

    def save_crops_raw(self):
        if self.img_bgr is None or not self.boxes:
            messagebox.showinfo("Info", "No crops to save. Run first.")
            return
        folder = filedialog.askdirectory(title="Save raw crops to folder")
        if not folder:
            return
        for i, (x1, y1, x2, y2) in enumerate(self.boxes):
            crop = self.img_bgr[y1:y2, x1:x2]
            cv2.imwrite(os.path.join(folder, f"digit_raw_{i:02d}.png"), crop)
        messagebox.showinfo("Saved", f"Saved {len(self.boxes)} raw crop(s) to:\n{folder}")

    def save_crops_28(self):
        if self.img_bgr is None or not self.boxes:
            messagebox.showinfo("Info", "No crops to save. Run first.")
            return
        folder = filedialog.askdirectory(title="Save 28x28 crops to folder")
        if not folder:
            return
        for i, (x1, y1, x2, y2) in enumerate(self.boxes):
            crop = self.img_bgr[y1:y2, x1:x2]
            d28 = to_28x28_centered(crop)
            cv2.imwrite(os.path.join(folder, f"digit_28x28_{i:02d}.png"), d28)
        messagebox.showinfo("Saved", f"Saved {len(self.boxes)} 28x28 crop(s) to:\n{folder}")

    # canvas 
    def _show(self, bgr):
        h, w = bgr.shape[:2]
        max_w, max_h = self.canvas_w, self.canvas_h
        scale = min(max_w / w, max_h / h, 1.0)
        new_w, new_h = int(w * scale), int(h * scale)
        disp = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        self.tk_img = ImageTk.PhotoImage(pil)
        self.canvas.delete("all")
        self.canvas.create_image(max_w // 2, max_h // 2, image=self.tk_img, anchor="center")

def main():
    root = tk.Tk()
    app = App(root)
    root.mainloop()

if __name__ == "__main__":
    main()
