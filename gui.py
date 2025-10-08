# import os
# import sys
# import cv2
# import tkinter as tk
# from tkinter import filedialog, messagebox, ttk
# from PIL import Image, ImageTk

# # Force reload of the module to ensure we're using the latest version
# import importlib
# import threshold_segmentation as seg
# importlib.reload(seg)  # This ensures we're using the most recent code


# class DigitSegmentationGUI:
#     def __init__(self, root: tk.Tk):
#         self.root = root
#         self.root.title("Digit Segmentation GUI - Using threshold_segmentation.py")
#         self.root.geometry("1100x680")

#         # State
#         self.image_bgr = None
#         self.bin_img = None
#         self.boxes = []
        
#         # Use the EXACT same parameters as threshold_segmentation.py
#         self.use_watershed = tk.BooleanVar(value=True)
#         self.use_adaptive = tk.BooleanVar(value=False)
#         self.use_simple_thresh = tk.BooleanVar(value=False)
#         self.thresh_val = tk.IntVar(value=128)
#         self.erode_iters = tk.IntVar(value=0)
#         self.trim_boxes = tk.BooleanVar(value=True)
#         self.no_contour_check = tk.BooleanVar(value=False)

#         self._tk_img_original = None
#         self._tk_img_bin = None

#         self._build_ui()

#     def _build_ui(self):
#         controls = ttk.Frame(self.root, padding=8)
#         controls.pack(side=tk.TOP, fill=tk.X)

#         # Basic controls
#         ttk.Button(controls, text="Open Image", command=self.open_image).pack(side=tk.LEFT, padx=4)
#         ttk.Button(controls, text="Capture Camera", command=self.capture_camera).pack(side=tk.LEFT, padx=4)
#         ttk.Button(controls, text="Segment", command=self.run_segmentation).pack(side=tk.LEFT, padx=12)
#         ttk.Button(controls, text="Save Crops", command=self.save_crops).pack(side=tk.LEFT, padx=4)
#         ttk.Button(controls, text="Clear", command=self.clear).pack(side=tk.LEFT, padx=4)

#         # Advanced options - match the threshold_segmentation.py parameters
#         advanced = ttk.LabelFrame(controls, text="Segmentation Options", padding=4)
#         advanced.pack(side=tk.LEFT, padx=20)

#         ttk.Checkbutton(advanced, text="Watershed", variable=self.use_watershed).pack(side=tk.LEFT, padx=4)
#         ttk.Checkbutton(advanced, text="Adaptive", variable=self.use_adaptive).pack(side=tk.LEFT, padx=4)
#         ttk.Checkbutton(advanced, text="Simple Thresh", variable=self.use_simple_thresh).pack(side=tk.LEFT, padx=4)
#         ttk.Checkbutton(advanced, text="Trim Boxes", variable=self.trim_boxes).pack(side=tk.LEFT, padx=4)
#         ttk.Checkbutton(advanced, text="No Contour Check", variable=self.no_contour_check).pack(side=tk.LEFT, padx=4)

#         thresh_frame = ttk.Frame(advanced)
#         thresh_frame.pack(side=tk.LEFT, padx=4)
#         ttk.Label(thresh_frame, text="Thresh:").pack(side=tk.LEFT)
#         ttk.Entry(thresh_frame, textvariable=self.thresh_val, width=4).pack(side=tk.LEFT, padx=2)

#         erode_frame = ttk.Frame(advanced)
#         erode_frame.pack(side=tk.LEFT, padx=4)
#         ttk.Label(erode_frame, text="Erode:").pack(side=tk.LEFT)
#         ttk.Entry(erode_frame, textvariable=self.erode_iters, width=2).pack(side=tk.LEFT, padx=2)

#         self.status_var = tk.StringVar(value="Load or capture an image to begin…")
#         ttk.Label(controls, textvariable=self.status_var).pack(side=tk.RIGHT)

#         body = ttk.Frame(self.root, padding=(8, 0, 8, 8))
#         body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

#         self.left_panel = self._make_panel(body, title="Original / Boxes")
#         self.right_panel = self._make_panel(body, title="Binarized")

#         self.left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))
#         self.right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(4, 0))

#     def _make_panel(self, parent, title=""):
#         frame = ttk.LabelFrame(parent, text=title, padding=8)
#         canvas = tk.Label(frame, bg="#111")
#         canvas.pack(fill=tk.BOTH, expand=True)
#         frame._canvas = canvas  
#         return frame

#     def open_image(self):
#         path = filedialog.askopenfilename(
#             title="Select image",
#             filetypes=[
#                 ("Images", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff"),
#                 ("All files", "*.*"),
#             ],
#         )
#         if not path:
#             return
#         img = cv2.imread(path)
#         if img is None:
#             messagebox.showerror("Error", f"Failed to read image: {path}")
#             return
#         self.image_bgr = img
#         self.boxes = []
#         self.bin_img = None
#         self.status_var.set(f"Loaded: {os.path.basename(path)}")
#         self._render()

#     def capture_camera(self):
#         cap = cv2.VideoCapture(0)
#         if not cap.isOpened():
#             messagebox.showerror("Camera Error", "Unable to open default camera (index 0).")
#             return
#         messagebox.showinfo("Camera", "Press SPACE to capture, ESC to cancel.")
#         captured = None
#         try:
#             while True:
#                 ok, frame = cap.read()
#                 if not ok:
#                     break
#                 cv2.imshow("Camera", frame)
#                 k = cv2.waitKey(1) & 0xFF
#                 if k == 27:
#                     captured = None
#                     break
#                 if k == 32:
#                     captured = frame.copy()
#                     break
#         finally:
#             cap.release()
#             cv2.destroyWindow("Camera")

#         if captured is not None:
#             self.image_bgr = captured
#             self.boxes = []
#             self.bin_img = None
#             self.status_var.set("Captured from camera")
#             self._render()

#     def run_segmentation(self):
#         if self.image_bgr is None:
#             messagebox.showwarning("No image", "Please open or capture an image first.")
#             return
        
#         try:
#             print("=" * 50)
#             print("RUNNING SEGMENTATION FROM threshold_segmentation.py")
#             print("=" * 50)
            
#             # Call the segmentation EXACTLY like in threshold_segmentation.py
#             boxes, bin_img = seg.segment_digits(
#                 self.image_bgr, 
#                 use_watershed=self.use_watershed.get(),
#                 use_simple_thresh=self.use_simple_thresh.get(),
#                 use_adaptive=self.use_adaptive.get(),
#                 thresh_val=self.thresh_val.get(),
#                 erode_iters=self.erode_iters.get(),
#                 erode_ksize=(2, 2)
#             )
            
#             print(f"Initial segmentation found {len(boxes)} boxes")
            
#             # Apply the same post-processing as in threshold_segmentation.py
#             if not self.no_contour_check.get():
#                 refined = []
#                 for b in boxes:
#                     x1, y1, x2, y2 = b
#                     roi = bin_img[y1:y2, x1:x2]
#                     cnts, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#                     if len(cnts) > 1:
#                         for c in cnts:
#                             xx, yy, ww, hh = cv2.boundingRect(c)
#                             refined.append((x1+xx, y1+yy, x1+xx+ww, y1+yy+hh))
#                     else:
#                         refined.append(b)
#                 boxes = sorted(refined, key=lambda r: r[0])
#                 print(f"After contour check: {len(boxes)} boxes")

#             # Apply tightening if requested
#             if self.trim_boxes.get():
#                 boxes = seg.tighten_boxes(bin_img, boxes, margin=2)
#                 print(f"After tightening: {len(boxes)} boxes")

#             # Final sort
#             boxes = sorted(boxes, key=lambda r: r[0])
            
#             print(f"FINAL RESULT: {len(boxes)} digit(s)")
#             for i, box in enumerate(boxes):
#                 print(f"  Box {i}: {box}")
#             print("=" * 50)
            
#         except Exception as e:
#             messagebox.showerror("Segmentation Error", str(e))
#             import traceback
#             traceback.print_exc()
#             return

#         self.boxes = boxes
#         self.bin_img = bin_img
#         self.status_var.set(f"Found {len(boxes)} digit(s)")
#         self._render()

#     def save_crops(self):
#         if self.image_bgr is None or not self.boxes:
#             messagebox.showwarning("Nothing to save", "Segment an image first.")
#             return
#         out_dir = filedialog.askdirectory(title="Choose output folder")
#         if not out_dir:
#             return
        
#         try:
#             # Use the save_crops function from threshold_segmentation.py
#             color_out = os.path.join(out_dir, "color")
#             bin_out = os.path.join(out_dir, "bin")
            
#             seg.save_crops(self.image_bgr, self.boxes, color_out)
#             seg.save_crops(cv2.cvtColor(self.bin_img, cv2.COLOR_GRAY2BGR), self.boxes, bin_out)
            
#             messagebox.showinfo("Saved", f"Saved {len(self.boxes)} crops to:\n{color_out}\n{bin_out}")
#         except Exception as e:
#             messagebox.showerror("Save Error", str(e))

#     def clear(self):
#         self.image_bgr = None
#         self.bin_img = None
#         self.boxes = []
#         self.status_var.set("Cleared. Load or capture an image…")
#         self._render()

#     def _render(self):
#         if self.image_bgr is None:
#             self._set_panel_image(self.left_panel, None)
#             self._set_panel_image(self.right_panel, None)
#             return

#         vis = self.image_bgr.copy()
#         if self.boxes:
#             # Use the draw_boxes function from threshold_segmentation.py
#             vis = seg.draw_boxes(vis, self.boxes)

#         self._set_panel_image(self.left_panel, vis[:, :, ::-1])  
        
#         if self.bin_img is not None:
#             rgb = cv2.cvtColor(self.bin_img, cv2.COLOR_GRAY2RGB)
#             self._set_panel_image(self.right_panel, rgb)
#         else:
#             self._set_panel_image(self.right_panel, None)

#     def _set_panel_image(self, panel: ttk.LabelFrame, img_rgb):
#         canvas: tk.Label = panel._canvas
#         if img_rgb is None:
#             canvas.config(image="", text="No image", fg="#bbb", font=("Segoe UI", 12))
#             return

#         panel.update_idletasks()
#         w = max(100, panel.winfo_width() - 16)
#         h = max(100, panel.winfo_height() - 48)

#         pil = Image.fromarray(img_rgb)
#         pil = self._fit_image(pil, (w, h))
#         tkimg = ImageTk.PhotoImage(pil)

#         canvas.config(image=tkimg, text="")
       
#         if panel is self.left_panel:
#             self._tk_img_original = tkimg
#         else:
#             self._tk_img_bin = tkimg

#     @staticmethod
#     def _fit_image(pil: Image.Image, max_size):
#         pil = pil.copy()
#         pil.thumbnail(max_size, Image.Resampling.LANCZOS)
#         return pil


# def main():
#     root = tk.Tk()
#     try:
#         style = ttk.Style(root)
#         if "vista" in style.theme_names():
#             style.theme_use("vista")
#         elif "clam" in style.theme_names():
#             style.theme_use("clam")
#     except Exception:
#         pass
#     app = DigitSegmentationGUI(root)
#     root.mainloop()


# if __name__ == "__main__":
#     main()