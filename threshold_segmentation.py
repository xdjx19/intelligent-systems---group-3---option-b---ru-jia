#threshold_segmentation.py - dwayne
#make sure cv is installed by running the following command in your terminal:
#pip install opencv-python numpy

#python threshold_segmentation.py sample1.png

import cv2
import numpy as np
import os
import sys

# --- thresholding methods ---

def binarize_otsu_inv(img_bgr):
    """
    Convert to grayscale and apply Otsu's threshold with inversion.
    Good when background/foreground vary in brightness.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return th

def binarize_simple_threshold(img_bgr, thresh_val=128, invert=True):
    """
    Convert to grayscale and apply a fixed global threshold.
    - thresh_val: intensity cutoff (0–255)
    - invert: if True, digits = white, background = black
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    flag = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, th = cv2.threshold(gray, thresh_val, 255, flag)
    return th


# --- morphological cleanup ---

def morph_close(bin_img, ksize=(3,3), iters=1):
    """Apply morphological closing to fill small gaps inside digits."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize)
    return cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel, iterations=iters)


# --- connected component analysis ---

def connected_components_boxes(bin_img, min_area=40, min_h=10, min_w=6, max_ar=6.0):
    """
    Find connected components and return bounding boxes that look like digits.
    Filters out too small/skinny/wide blobs.
    """
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bin_img, connectivity=8)
    boxes = []
    for i in range(1, num):  # skip background
        x, y, w, h, area = stats[i]
        if area < min_area or h < min_h or w < min_w:
            continue
        ar = h / (w + 1e-6)  # aspect ratio
        if ar > max_ar:
            continue
        boxes.append((x, y, x + w, y + h))
    boxes = _merge_overlaps(boxes, iou_thresh=0.3)
    boxes.sort(key=lambda b: b[0])  # left-to-right order
    return boxes

def _merge_overlaps(boxes, iou_thresh=0.3):
    """Merge overlapping bounding boxes based on IoU threshold."""
    def iou(a, b):
        x1, y1 = max(a[0], b[0]), max(a[1], b[1])
        x2, y2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        areaA = (a[2] - a[0]) * (a[3] - a[1])
        areaB = (b[2] - b[0]) * (b[3] - b[1])
        return inter / (areaA + areaB - inter + 1e-6)

    kept = []
    for b in boxes:
        merged = False
        for i, k in enumerate(kept):
            if iou(b, k) > iou_thresh:
                kept[i] = (min(b[0], k[0]), min(b[1], k[1]),
                           max(b[2], k[2]), max(b[3], k[3]))
                merged = True
                break
        if not merged:
            kept.append(b)
    return kept


# --- watershed split for touching digits ---

def watershed_split(bin_img, bbox):
    """Split joined digits inside a bounding box using watershed segmentation."""
    x1, y1, x2, y2 = bbox
    roi = bin_img[y1:y2, x1:x2]
    if roi.size == 0:
        return [bbox]

    dist = cv2.distanceTransform(roi, cv2.DIST_L2, 5)
    if dist.max() <= 0:
        return [bbox]
    _, sure_fg = cv2.threshold(dist, 0.5 * dist.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)
    unknown = cv2.subtract(roi, sure_fg)

    num, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    roi3 = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
    cv2.watershed(roi3, markers)

    subboxes = []
    for label in range(2, num + 1):
        ys, xs = np.where(markers == label)
        if len(xs) == 0:
            continue
        subboxes.append((x1 + int(xs.min()), y1 + int(ys.min()),
                         x1 + int(xs.max()) + 1, y1 + int(ys.max()) + 1))
    return subboxes if subboxes else [bbox]


# --- main segmentation ---

def segment_digits(image_bgr, use_watershed=True, use_simple_thresh=False, thresh_val=128):
    """
    Segment digits from an image.
    - use_watershed: try to split merged digits
    - use_simple_thresh: use fixed thresholding instead of Otsu
    - thresh_val: cutoff for simple thresholding
    """
    if use_simple_thresh:
        bin_img = binarize_simple_threshold(image_bgr, thresh_val=thresh_val)
    else:
        bin_img = binarize_otsu_inv(image_bgr)

    bin_img = morph_close(bin_img, (3, 3), 1)

    boxes = connected_components_boxes(bin_img)

    if use_watershed:
        refined = []
        for b in boxes:
            x1, y1, x2, y2 = b
            w, h = (x2 - x1), (y2 - y1)
            if w > 1.4 * h:  # wide boxes may contain 2+ digits
                parts = watershed_split(bin_img, b)
                refined.extend(parts)
            else:
                refined.append(b)
        boxes = sorted(refined, key=lambda r: r[0])

    return boxes, bin_img


# --- utils ---

def save_crops(img, boxes, out_dir):
    """Save cropped digit images into output directory."""
    os.makedirs(out_dir, exist_ok=True)
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        crop = img[y1:y2, x1:x2]
        cv2.imwrite(os.path.join(out_dir, f"digit_{i:02d}.png"), crop)

def draw_boxes(img, boxes):
    """Draw green bounding boxes around digits."""
    vis = img.copy()
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return vis


# --- entry point ---

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python segment_digits.py <image_path> [--no-watershed] [--simple-thresh val] [--save out_dir]")
        sys.exit(1)

    img_path = sys.argv[1]
    use_ws = "--no-watershed" not in sys.argv

    # check if simple thresholding requested
    if "--simple-thresh" in sys.argv:
        idx = sys.argv.index("--simple-thresh")
        thresh_val = int(sys.argv[idx + 1]) if idx + 1 < len(sys.argv) else 128
        use_simple = True
    else:
        thresh_val = 128
        use_simple = False

    save_idx = sys.argv.index("--save") + 1 if "--save" in sys.argv else None
    out_dir = sys.argv[save_idx] if save_idx else None

    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(img_path)

    boxes, bin_img = segment_digits(img, use_watershed=use_ws,
                                    use_simple_thresh=use_simple, thresh_val=thresh_val)
    vis = draw_boxes(img, boxes)

    print(f"Found {len(boxes)} digit(s). Boxes (x1,y1,x2,y2):\n{boxes}")

    cv2.imshow("Binarized", bin_img)
    cv2.imshow("Digits (boxed)", vis)
    cv2.waitKey(0)

    if out_dir:
        save_crops(cv2.cvtColor(bin_img, cv2.COLOR_GRAY2BGR), boxes, out_dir)
        print(f"Crops saved to: {out_dir}")

print("Running segmentation...")
print("Arguments:", sys.argv)