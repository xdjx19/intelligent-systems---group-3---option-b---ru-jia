import os
import cv2
import numpy as np
import ast
import tkinter as tk
from pathlib import Path
import types
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk

# Local imports
import threshold_segmentation as seg


try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None

TORCH_AVAILABLE = torch is not None

_SMALLCNN_CLASS = None


def _load_smallcnn_class():
    global _SMALLCNN_CLASS
    if _SMALLCNN_CLASS is not None:
        return _SMALLCNN_CLASS
    if not TORCH_AVAILABLE:
        return None
    cnn_path = Path(__file__).with_name('CNN.py')
    if not cnn_path.is_file():
        raise FileNotFoundError('CNN.py')
    source = cnn_path.read_text(encoding='utf-8')
    module_ast = ast.parse(source, filename=str(cnn_path))
    target = None
    for node in module_ast.body:
        if isinstance(node, ast.ClassDef) and node.name == 'SmallCNN':
            target = node
            break
    if target is None:
        raise RuntimeError('SmallCNN class not found in CNN.py')
    module = types.ModuleType('cnn_smallcnn')
    module.__dict__['torch'] = torch
    module.__dict__['nn'] = nn
    exec(compile(ast.Module(body=[target], type_ignores=[]), str(cnn_path), 'exec'), module.__dict__)
    _SMALLCNN_CLASS = module.SmallCNN
    return _SMALLCNN_CLASS


class DigitRecognizer:
    def __init__(self, weights_path: str = 'cnn_mnist_trained.pt'):
        self.weights_path = weights_path
        self.device = 'cuda' if TORCH_AVAILABLE and torch.cuda.is_available() else 'cpu'
        self.model = None

    def set_weights_path(self, path: str):
        self.weights_path = path
        self.model = None

    def ensure_loaded(self):
        if not TORCH_AVAILABLE:
            raise RuntimeError('PyTorch is not installed. Install torch to enable recognition.')
        if self.model is not None:
            return
        model_cls = _load_smallcnn_class()
        if model_cls is None:
            raise RuntimeError('CNN architecture unavailable (torch import failed).')
        if not os.path.isfile(self.weights_path):
            raise FileNotFoundError(self.weights_path)
        state = torch.load(self.weights_path, map_location=self.device)
        num_classes = state['fc2.weight'].shape[0] if 'fc2.weight' in state else 10
        self.model = model_cls(num_classes=num_classes).to(self.device)
        self.model.load_state_dict(state)
        self.model.eval()

    def predict_boxes(self, bin_img, boxes):
        self.ensure_loaded()
        tensors = []
        kept_boxes = []
        for box in boxes:
            tensor = self._prepare_tensor(bin_img, box)
            if tensor is not None:
                tensors.append(tensor)
                kept_boxes.append(box)
        if not tensors:
            return []
        batch = torch.cat(tensors, dim=0).to(self.device)
        with torch.no_grad():
            logits = self.model(batch)
            probs = torch.softmax(logits, dim=1)
            confs, labels = probs.max(dim=1)
        results = []
        for box, label, conf in zip(kept_boxes, labels.cpu().tolist(), confs.cpu().tolist()):
            results.append((box, label, conf))
        return results

    @staticmethod
    def _prepare_tensor(bin_img, box):
        if not TORCH_AVAILABLE:
            return None
        x1, y1, x2, y2 = box
        roi = bin_img[y1:y2, x1:x2]
        if roi.size == 0:
            return None
        if roi.ndim == 3:
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        if roi.max() == 0:
            return None
        h, w = roi.shape[:2]
        size = max(h, w)
        padded = np.zeros((size, size), dtype=np.uint8)
        y_offset = (size - h) // 2
        x_offset = (size - w) // 2
        padded[y_offset:y_offset + h, x_offset:x_offset + w] = roi
        resized = cv2.resize(padded, (28, 28), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(resized.astype(np.float32) / 255.0)
        return tensor.unsqueeze(0).unsqueeze(0)

class DigitSegmentationGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Digit Segmentation GUI")
        self.root.geometry("1100x680")

        # State
        self.image_bgr = None  # OpenCV BGR image
        self.bin_img = None    # Binary image (uint8)
        self.boxes = []
        self.use_watershed = tk.BooleanVar(value=True)

        self.recognizer = DigitRecognizer()
        self.predictions = []

        self._tk_img_original = None
        self._tk_img_bin = None

        self._build_ui()

    def _build_ui(self):
       
        controls = ttk.Frame(self.root, padding=8)
        controls.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(controls, text="Open Image", command=self.open_image).pack(side=tk.LEFT, padx=4)
        ttk.Button(controls, text="Capture Camera", command=self.capture_camera).pack(side=tk.LEFT, padx=4)

        ttk.Checkbutton(controls, text="Use Watershed split", variable=self.use_watershed).pack(side=tk.LEFT, padx=12)

        ttk.Button(controls, text="Segment", command=self.run_segmentation).pack(side=tk.LEFT, padx=12)
        ttk.Button(controls, text="Recognize", command=self.recognize_digits).pack(side=tk.LEFT, padx=12)
        ttk.Button(controls, text="Save Crops", command=self.save_crops).pack(side=tk.LEFT, padx=4)
        ttk.Button(controls, text="Clear", command=self.clear).pack(side=tk.LEFT, padx=4)

        self.status_var = tk.StringVar(value="Load or capture an image to begin...")
        ttk.Label(controls, textvariable=self.status_var).pack(side=tk.RIGHT)

        
        body = ttk.Frame(self.root, padding=(8, 0, 8, 8))
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.left_panel = self._make_panel(body, title="Original / Boxes")
        self.right_panel = self._make_panel(body, title="Binarized")

        self.left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))
        self.right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(4, 0))

    def _make_panel(self, parent, title=""):
        frame = ttk.LabelFrame(parent, text=title, padding=8)
        canvas = tk.Label(frame, bg="#111")
        canvas.pack(fill=tk.BOTH, expand=True)
        frame._canvas = canvas  
        return frame

    
    def open_image(self):
        path = filedialog.askopenfilename(
            title="Select image",
            filetypes=[
                ("Images", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Error", f"Failed to read image: {path}")
            return
        self.image_bgr = img
        self.boxes = []
        self.bin_img = None
        self.predictions = []
        self.status_var.set(f"Loaded: {os.path.basename(path)}")
        self._render()

    def capture_camera(self):
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Camera Error", "Unable to open default camera (index 0).")
            return
        messagebox.showinfo("Camera", "Press SPACE to capture, ESC to cancel.")
        captured = None
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                cv2.imshow("Camera", frame)
                k = cv2.waitKey(1) & 0xFF
                if k == 27:  # ESC
                    captured = None
                    break
                if k == 32:  # SPACE
                    captured = frame.copy()
                    break
        finally:



















            
            cap.release()
            cv2.destroyWindow("Camera")

        if captured is not None:
            self.image_bgr = captured
            self.boxes = []
            self.bin_img = None
            self.predictions = []
            self.status_var.set("Captured from camera")
            self._render()

    def run_segmentation(self):
        if self.image_bgr is None:
            messagebox.showwarning("No image", "Please open or capture an image first.")
            return
        try:
            boxes, bin_img = seg.segment_digits(
                self.image_bgr, use_watershed=self.use_watershed.get()
            )
        except Exception as e:
            messagebox.showerror("Segmentation Error", str(e))
            return

        self.boxes = boxes
        self.bin_img = bin_img
        self.status_var.set(f"Found {len(boxes)} digit(s)")
        self.predictions = []
        self._render()

    def recognize_digits(self):
        if self.image_bgr is None or not self.boxes:
            messagebox.showwarning("No digits", "Segment an image first.")
            return
        if self.bin_img is None:
            messagebox.showwarning("No segmentation", "Run segmentation before recognition.")
            return
        try:
            predictions = self.recognizer.predict_boxes(self.bin_img, self.boxes)
        except FileNotFoundError:
            path = filedialog.askopenfilename(
                title="Select CNN weight file",
                filetypes=[("PyTorch weights", "*.pt;*.pth"), ("All files", "*.*")]
            )
            if not path:
                return
            self.recognizer.set_weights_path(path)
            try:
                predictions = self.recognizer.predict_boxes(self.bin_img, self.boxes)
            except Exception as err:
                messagebox.showerror("Recognition Error", str(err))
                return
        except RuntimeError as err:
            messagebox.showerror("Recognition Error", str(err))
            return
        except Exception as err:
            messagebox.showerror("Recognition Error", str(err))
            return
        if not predictions:
            messagebox.showinfo("Recognition", "No digit crops available for recognition.")
            return
        self.predictions = predictions
        sequence = ''.join(str(label) for _, label, _ in predictions)
        avg_conf = sum(conf for _, _, conf in predictions) / len(predictions)
        self.status_var.set(f"Predicted digits: {sequence} (avg {avg_conf * 100:.1f}%)")
        self._render()

    def save_crops(self):
        if self.image_bgr is None or not self.boxes:
            messagebox.showwarning("Nothing to save", "Segment an image first.")
            return
        out_dir = filedialog.askdirectory(title="Choose output folder")
        if not out_dir:
            return
        
        try:
            color_bin = cv2.cvtColor(self.bin_img, cv2.COLOR_GRAY2BGR)
            seg.save_crops(color_bin, self.boxes, out_dir)
            messagebox.showinfo("Saved", f"Saved {len(self.boxes)} crops to:\n{out_dir}")
        except Exception as e:
            messagebox.showerror("Save Error", str(e))

    def clear(self):
        self.image_bgr = None
        self.bin_img = None
        self.boxes = []
        self.predictions = []
        self.status_var.set("Cleared. Load or capture an image...")
        self._render()

    
    def _render(self):
        
        if self.image_bgr is None:
            self._set_panel_image(self.left_panel, None)
            self._set_panel_image(self.right_panel, None)
            return

        vis = self.image_bgr.copy()
        if self.boxes:
            vis = seg.draw_boxes(vis, self.boxes)
            if self.predictions:
                for box, label, conf in self.predictions:
                    x1, y1, _, _ = map(int, box)
                    text = f"{label} ({conf * 100:.0f}%)"
                    y_pos = max(y1 - 8, 18)
                    cv2.putText(
                        vis,
                        text,
                        (x1, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )

        self._set_panel_image(self.left_panel, vis[:, :, ::-1])  
        if self.bin_img is not None:
            rgb = cv2.cvtColor(self.bin_img, cv2.COLOR_GRAY2RGB)
            self._set_panel_image(self.right_panel, rgb)
        else:
            self._set_panel_image(self.right_panel, None)

    def _set_panel_image(self, panel: ttk.LabelFrame, img_rgb):
        canvas: tk.Label = panel._canvas
        if img_rgb is None:
            canvas.config(image="", text="No image", fg="#bbb", font=("Segoe UI", 12))
            return

        
        panel.update_idletasks()
        w = max(100, panel.winfo_width() - 16)
        h = max(100, panel.winfo_height() - 48)

        pil = Image.fromarray(img_rgb)
        pil = self._fit_image(pil, (w, h))
        tkimg = ImageTk.PhotoImage(pil)

        canvas.config(image=tkimg, text="")
       
        if panel is self.left_panel:
            self._tk_img_original = tkimg
        else:
            self._tk_img_bin = tkimg

    @staticmethod
    def _fit_image(pil: Image.Image, max_size):
        pil = pil.copy()
        pil.thumbnail(max_size, Image.Resampling.LANCZOS)
        return pil


def main():
    root = tk.Tk()
    try:
        
        style = ttk.Style(root)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        elif "clam" in style.theme_names():
            style.theme_use("clam")
    except Exception:
        pass
    app = DigitSegmentationGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

