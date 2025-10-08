# threshold_segmentation.py - CLI version (no Tkinter GUI)
# dwayne - digit segmentation with thresholding

import os
import sys
import cv2
import numpy as np


# -------------------------------
# Segmentation + Utility Functions
# -------------------------------

def segment_digits(
    img_bgr,
    use_watershed=True,
    use_simple_thresh=False,
    use_adaptive=False,
    thresh_val=128,
    erode_iters=0,
    erode_ksize=(2, 2)
):
    """Perform digit segmentation on an input BGR image."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Thresholding
    if use_adaptive:
        bin_img = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
    elif use_simple_thresh:
        _, bin_img = cv2.threshold(
            gray, thresh_val, 255, cv2.THRESH_BINARY_INV
        )
    else:
        # Otsu fallback
        _, bin_img = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

    # Erosion (optional)
    if erode_iters > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, erode_ksize)
        bin_img = cv2.erode(bin_img, kernel, iterations=erode_iters)

    # Find contours
    cnts, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w > 3 and h > 3:  # filter noise
            boxes.append((x, y, x + w, y + h))

    # Optional: watershed can refine segmentation (disabled by default for speed)
    if use_watershed:
        kernel = np.ones((3, 3), np.uint8)
        sure_bg = cv2.dilate(bin_img, kernel, iterations=3)

        dist_transform = cv2.distanceTransform(bin_img, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)

        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        cv2.watershed(img_bgr.copy(), markers)

    return sorted(boxes, key=lambda r: r[0]), bin_img


def tighten_boxes(bin_img, boxes, margin=2):
    """Tighten bounding boxes to better fit digit shapes."""
    tightened = []
    for (x1, y1, x2, y2) in boxes:
        roi = bin_img[y1:y2, x1:x2]
        cnts, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            xx, yy, ww, hh = cv2.boundingRect(max(cnts, key=cv2.contourArea))
            tightened.append((x1+xx-margin, y1+yy-margin, x1+xx+ww+margin, y1+yy+hh+margin))
        else:
            tightened.append((x1, y1, x2, y2))
    return tightened


def draw_boxes(img_bgr, boxes, color=(0, 255, 0), thickness=2):
    """Draw bounding boxes on a copy of the image."""
    out = img_bgr.copy()
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
    return out


def save_crops(img_bgr, boxes, out_dir):
    """Save cropped digit images into output folder."""
    os.makedirs(out_dir, exist_ok=True)
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        crop = img_bgr[y1:y2, x1:x2]
        out_path = os.path.join(out_dir, f"crop_{i:03d}.png")
        cv2.imwrite(out_path, crop)


# -------------------------------
# CLI Entry Point
# -------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python threshold_segmentation.py <image_path> [output_dir]")
        sys.exit(1)

    image_path = sys.argv[1]
    save_dir = sys.argv[2] if len(sys.argv) > 2 else None

    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Could not read image: {image_path}")
        sys.exit(1)

    boxes, bin_img = segment_digits(img)

    # Refine + tighten
    boxes = tighten_boxes(bin_img, boxes, margin=2)

    print(f"FINAL RESULT: {len(boxes)} digit(s)")
    for i, b in enumerate(boxes):
        print(f"  Box {i}: {b}")

    # Draw boxes and show
    vis = draw_boxes(img, boxes)
    cv2.imshow("Segmentation Result", vis)
    cv2.imshow("Binarized", bin_img)
    print("[INFO] Press any key in the image window to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save crops if requested
    if save_dir:
        color_out = os.path.join(save_dir, "color")
        bin_out = os.path.join(save_dir, "bin")
        save_crops(img, boxes, color_out)
        save_crops(cv2.cvtColor(bin_img, cv2.COLOR_GRAY2BGR), boxes, bin_out)
        print(f"[INFO] Saved {len(boxes)} crops to: {color_out}, {bin_out}")


if __name__ == "__main__":
    main()
