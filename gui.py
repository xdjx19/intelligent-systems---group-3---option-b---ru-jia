import os
import sys
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk

# Local imports
import segment_digits as seg


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
        ttk.Button(controls, text="Save Crops", command=self.save_crops).pack(side=tk.LEFT, padx=4)
        ttk.Button(controls, text="Clear", command=self.clear).pack(side=tk.LEFT, padx=4)

        self.status_var = tk.StringVar(value="Load or capture an image to begin…")
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
            self.status_var.set("Captured from camera")
            self._render()

    def run_segmentation(self):
        if self.image_bgr is None:
            messagebox.showwarning("No image", "Please open or capture an image first.")
            return
        try:
            boxes, bin_img = seg.segment_digits(self.image_bgr, use_watershed=self.use_watershed.get())
        except Exception as e:
            messagebox.showerror("Segmentation Error", str(e))
            return

        self.boxes = boxes
        self.bin_img = bin_img
        self.status_var.set(f"Found {len(boxes)} digit(s)")
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
        self.status_var.set("Cleared. Load or capture an image…")
        self._render()

    
    def _render(self):
        
        if self.image_bgr is None:
            self._set_panel_image(self.left_panel, None)
            self._set_panel_image(self.right_panel, None)
            return

        vis = self.image_bgr.copy()
        if self.boxes:
            vis = seg.draw_boxes(vis, self.boxes)

        
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

