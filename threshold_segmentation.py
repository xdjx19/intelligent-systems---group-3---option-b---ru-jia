# threshold_segmentation.py - dwayne
# make sure cv2 is installed:
# pip install opencv-python numpy
#
# Usage:
#   python threshold_segmentation.py sample1.png
#   python threshold_segmentation.py sample1.png --simple-thresh 140
#   python threshold_segmentation.py sample1.png --no-watershed

import cv2
import numpy as np
import os
import sys

# --- thresholding methods ---

def binarize_otsu_inv(img_bgr):
    """Convert to grayscale and apply Otsu's threshold with inversion."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return th

def binarize_simple_threshold(img_bgr, thresh_val=128, invert=True):
    """Convert to grayscale and apply a fixed global threshold."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    flag = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, th = cv2.threshold(gray, thresh_val, 255, flag)
    return th


# --- morphology helpers ---

def morph_close(bin_img, ksize=(3,3), iters=1):
    """Closing = dilation then erosion; fills small gaps in strokes."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize)
    return cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel, iterations=iters)

def morph_dilate(bin_img, ksize=(3,3), iters=1):
    """Light dilation to thicken thin strokes (helps contour fallback)."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize)
    return cv2.dilate(bin_img, kernel, iterations=iters)


# --- connected components (kept, used as backup if needed) ---

def connected_components_boxes(bin_img, min_area=40, min_h=10, min_w=6, max_ar=6.0):
    """Find connected components and return bounding boxes that look like digits."""
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bin_img, connectivity=8)
    boxes = []
    for i in range(1, num):  # skip background
        x, y, w, h, area = stats[i]
        if area < min_area or h < min_h or w < min_w:
            continue
        ar = h / (w + 1e-6)
        if ar > max_ar:
            continue
        boxes.append((x, y, x + w, y + h))
    boxes = _merge_overlaps(boxes, iou_thresh=0.3)
    boxes.sort(key=lambda b: b[0])
    return boxes

def _merge_overlaps(boxes, iou_thresh=0.3):
    """Merge overlapping bounding boxes based on IoU."""
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


# --- watershed split for touching digits (kept) ---

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


# --- robust projection splitter (NEW) ---

def split_digits_projection(bin_img, min_width=20, smooth_w=25, gap_rel=0.18, gap_min_w=3):
    """
    Split digits by vertical projection using *valleys* in a smoothed column-sum profile.
    - bin_img: binary image (digits = white, background = black)
    - min_width: minimum width of each digit region
    - smooth_w: moving-average window for smoothing the projection (odd recommended)
    - gap_rel: cut when valley <= gap_rel * max(profile)  (lower => stricter splits)
    - gap_min_w: minimum width of a gap to be considered a separator
    Returns: list of (x1, y1, x2, y2)
    """
    H, W = bin_img.shape[:2]
    # Column-wise sum of white pixels
    proj = (bin_img > 0).sum(axis=0).astype(np.float32)

    # Smooth to remove spurious oscillations
    k = max(3, int(smooth_w))
    if k % 2 == 0: k += 1
    kernel = np.ones(k, dtype=np.float32) / k
    smooth = np.convolve(proj, kernel, mode='same')

    # A "gap" is where the profile dips near background
    thresh = gap_rel * (smooth.max() if smooth.max() > 0 else 1.0)
    gap_mask = smooth <= thresh

    # Merge tiny toggles in gap_mask by closing small holes
    # (one pass of dilation then erosion in 1D)
    gap_mask = _binary_1d_close(gap_mask.astype(np.uint8), size=gap_min_w)

    # Find contiguous gap runs
    boxes = []
    starts = []
    in_fg = False
    for x in range(W):
        if not in_fg and not gap_mask[x]:
            in_fg = True
            starts.append(x)
        if in_fg and (x == W - 1 or gap_mask[x] and (x+1 < W and gap_mask[x+1])):
            # end before the gap starts
            end = x if gap_mask[x] else x + 1
            if starts:
                s = starts.pop(0)
                if end - s >= min_width:
                    boxes.append((s, 0, end, H))
            in_fg = False

    # If nothing reasonable found, return a single full-width box (handled by fallback)
    return boxes

def _binary_1d_close(arr01, size=3):
    """Simple 1D binary closing for a boolean array (numpy), using min/max filters."""
    size = max(1, int(size))
    # Dilation
    dil = np.copy(arr01)
    for i in range(len(arr01)):
        left = max(0, i - size + 1)
        if arr01[left:i+1].max() == 1:
            dil[i] = 1
    # Erosion
    ero = np.copy(dil)
    for i in range(len(dil)):
        left = max(0, i - size + 1)
        if dil[left:i+1].min() == 1:
            ero[i] = 1
        else:
            ero[i] = 0
    return ero


# --- contour-based fallback (NEW) ---

def split_digits_contours_fallback(bin_img, min_area=80, dilate_ksize=3, dilate_iters=1):
    """
    Fallback: slightly dilate, then take external contours as digit boxes.
    Good when projection fails due to touching strokes or no clear valleys.
    """
    H, W = bin_img.shape[:2]
    thick = morph_dilate(bin_img, (dilate_ksize, dilate_ksize), dilate_iters)
    contours, _ = cv2.findContours(thick, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h < min_area:
            continue
        boxes.append((x, y, x + w, y + h))

    if not boxes:
        return [(0, 0, W, H)]  # worst case: whole line

    boxes.sort(key=lambda b: b[0])
    return boxes


# --- main segmentation ---

def segment_digits(image_bgr, use_watershed=True, use_simple_thresh=False, thresh_val=128):
    """Segment digits from an image."""
    # 1) Binarise (digits white on black)
    if use_simple_thresh:
        bin_img = binarize_simple_threshold(image_bgr, thresh_val=thresh_val)
    else:
        bin_img = binarize_otsu_inv(image_bgr)

    # 2) Clean tiny gaps inside strokes
    bin_img = morph_close(bin_img, (3, 3), 1)

    # 3) Try robust projection split first
    boxes = split_digits_projection(bin_img, min_width=20, smooth_w=25, gap_rel=0.18, gap_min_w=3)

    # 4) If projection couldn't split well, try contour fallback (with light dilation)
    if len(boxes) <= 1:
        boxes = split_digits_contours_fallback(bin_img, min_area=80, dilate_ksize=3, dilate_iters=1)

    # 5) Optional: watershed refinement for very wide boxes
    if use_watershed and len(boxes) > 0:
        refined = []
        for b in boxes:
            x1, y1, x2, y2 = b
            w, h = (x2 - x1), (y2 - y1)
            if w > 1.4 * h:
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
        print("Usage: python threshold_segmentation.py <image_path> [--no-watershed] [--simple-thresh val] [--save out_dir]")
        sys.exit(1)

    img_path = sys.argv[1]
    use_ws = "--no-watershed" not in sys.argv

    # optional fixed thresholding
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