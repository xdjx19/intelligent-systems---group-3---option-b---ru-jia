# threshold_segmentation.py - dwayne (improved tighter & smarter splitting)
# make sure cv2 is installed:
# pip install opencv-python numpy
#
# Usage examples:
#   python threshold_segmentation.py sample1.png
#   python threshold_segmentation.py sample1.png --simple-thresh 140 --adaptive
#   python threshold_segmentation.py sample1.png --trim --erode 1 --save out_dir
#
# Key improvements in this file:
# - argparse CLI (clean flags)
# - optional adaptive thresholding (--adaptive)
# - optional controlled erosion/dilation to separate touching digits (--erode N)
# - projection-based smart-splitting inside wide boxes (find actual valley rather than naive half-split)
# - tightening boxes to foreground pixels (like the 7)
# - contour refinement toggle (--no-contour-check)
# - watershed fallback remains for difficult overlaps
# - comments preserved / expanded

import cv2
import numpy as np
import os
import sys
import argparse

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

def binarize_adaptive(img_bgr, block_size=21, C=10, invert=True):
    """Adaptive Gaussian threshold (better under uneven lighting)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    flag = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               flag, block_size, C)
    return th

# --- morphology helpers ---

def morph_close(bin_img, ksize=(3,3), iters=1):
    """Closing = dilation then erosion; fills small gaps in strokes."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize)
    return cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel, iterations=iters)

def morph_open(bin_img, ksize=(2,2), iters=1):
    """Opening = erosion then dilation; removes small specks/noise."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize)
    return cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel, iterations=iters)

def morph_erode(bin_img, ksize=(2,2), iters=1):
    """Erode to shrink strokes (help separate touching digits)."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize)
    return cv2.erode(bin_img, kernel, iterations=iters)

def morph_dilate(bin_img, ksize=(3,3), iters=1):
    """Light dilation to thicken thin strokes (helps contour fallback)."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize)
    return cv2.dilate(bin_img, kernel, iterations=iters)

# --- connected components (backup method) ---

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

# --- projection split ---

def split_digits_projection(bin_img, min_width=20, smooth_w=25, gap_rel=0.18, gap_min_w=3):
    """Split digits by vertical projection profile valleys."""
    H, W = bin_img.shape[:2]
    proj = (bin_img > 0).sum(axis=0).astype(np.float32)

    k = max(3, int(smooth_w))
    if k % 2 == 0: k += 1
    kernel = np.ones(k, dtype=np.float32) / k
    smooth = np.convolve(proj, kernel, mode='same')

    thresh = gap_rel * (smooth.max() if smooth.max() > 0 else 1.0)
    gap_mask = smooth <= thresh
    gap_mask = _binary_1d_close(gap_mask.astype(np.uint8), size=gap_min_w)

    boxes = []
    starts = []
    in_fg = False
    for x in range(W):
        if not in_fg and not gap_mask[x]:
            in_fg = True
            starts.append(x)
        if in_fg and (x == W - 1 or (gap_mask[x] and (x+1 < W and gap_mask[x+1]))):
            end = x if gap_mask[x] else x + 1
            if starts:
                s = starts.pop(0)
                if end - s >= min_width:
                    boxes.append((s, 0, end, H))
            in_fg = False
    return boxes

def _binary_1d_close(arr01, size=3):
    """Simple 1D binary closing for 1D projection gaps."""
    size = max(1, int(size))
    dil = np.copy(arr01)
    for i in range(len(arr01)):
        left = max(0, i - size + 1)
        if arr01[left:i+1].max() == 1:
            dil[i] = 1
    ero = np.copy(dil)
    for i in range(len(dil)):
        left = max(0, i - size + 1)
        if dil[left:i+1].min() == 1:
            ero[i] = 1
        else:
            ero[i] = 0
    return ero

# --- contour fallback ---

def split_digits_contours_fallback(bin_img, min_area=80, dilate_ksize=3, dilate_iters=1):
    """Fallback: dilate slightly, then take contours as digit boxes."""
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
        return [(0, 0, W, H)]
    boxes.sort(key=lambda b: b[0])
    return boxes

# --- NEW: tighten bounding boxes ---

def tighten_boxes(bin_img, boxes, margin=2):
    """
    Shrink bounding boxes to the actual digit strokes inside.
    Adds a small margin so digits aren't cut off.
    """
    H, W = bin_img.shape[:2]
    refined = []
    for (x1, y1, x2, y2) in boxes:
        roi = bin_img[y1:y2, x1:x2]
        ys, xs = np.where(roi > 0)
        if len(xs) == 0:
            refined.append((x1, y1, x2, y2))
            continue
        nx1 = max(0, x1 + xs.min() - margin)
        ny1 = max(0, y1 + ys.min() - margin)
        nx2 = min(W, x1 + xs.max() + margin)
        ny2 = min(H, y1 + ys.max() + margin)
        refined.append((nx1, ny1, nx2, ny2))
    return refined

# --- SMART splitting inside a wide box (find valley) ---

def smart_split_box(bin_img, bbox, min_gap_width=3, smooth_w=9, gap_rel=0.18):
    """
    Given a bounding box that is likely to contain multiple digits,
    look for a vertical valley in the column projection inside the box
    and split at the best valley (return list of boxes).
    If no good valley found, fall back to splitting in halves.
    """
    x1, y1, x2, y2 = bbox
    roi = bin_img[y1:y2, x1:x2]
    H, W = roi.shape[:2]
    if W <= 1:
        return [bbox]

    proj = (roi > 0).sum(axis=0).astype(np.float32)

    # smooth projection
    k = max(3, int(smooth_w))
    if k % 2 == 0: k += 1
    kernel = np.ones(k, dtype=np.float32) / k
    smooth = np.convolve(proj, kernel, mode='same')

    # valley threshold relative to local max
    thresh = gap_rel * (smooth.max() if smooth.max() > 0 else 1.0)
    valley_mask = smooth <= thresh

    # find contiguous valley runs (where valley_mask==1)
    runs = []
    i = 0
    while i < W:
        if valley_mask[i]:
            j = i
            while j+1 < W and valley_mask[j+1]:
                j += 1
            runs.append((i, j))
            i = j + 1
        else:
            i += 1

    # choose the best run as the one closest to center and wide enough
    if runs:
        center = W / 2.0
        best = None
        best_score = 1e9
        for (s, e) in runs:
            width = e - s + 1
            if width < min_gap_width:
                continue
            # prefer valley near center and with deeper dip
            valley_depth = smooth[s:e+1].min()
            score = abs((s+e)/2.0 - center) - 0.5 * (smooth.max() - valley_depth)
            if score < best_score:
                best_score = score
                best = (s, e)
        if best is not None:
            mid = (best[0] + best[1]) // 2
            if mid > 0 and mid < W-1:
                left = (x1, y1, x1 + mid, y2)
                right = (x1 + mid, y1, x2, y2)
                # don't create tiny boxes
                if (left[2] - left[0]) >= 6 and (right[2] - right[0]) >= 6:
                    return [left, right]

    # fallback: split by connected component contours inside roi
    cnts, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) >= 2:
        sub = []
        for c in cnts:
            xx, yy, ww, hh = cv2.boundingRect(c)
            sub.append((x1 + xx, y1 + yy, x1 + xx + ww, y1 + yy + hh))
        sub = sorted(sub, key=lambda b: b[0])
        return sub

    # last resort: equal half split
    half = W // 2
    if half < 6:
        return [bbox]
    left = (x1, y1, x1 + half, y2)
    right = (x1 + half, y1, x2, y2)
    return [left, right]

# --- main segmentation ---

def segment_digits(image_bgr, use_watershed=True, use_simple_thresh=False,
                   use_adaptive=False, thresh_val=128,
                   erode_iters=0, erode_ksize=(2,2)):
    """Segment digits from an image. Returns boxes (x1,y1,x2,y2) and bin_img."""
    # 1) Binarise (digits white on black)
    if use_adaptive:
        bin_img = binarize_adaptive(image_bgr)
    elif use_simple_thresh:
        bin_img = binarize_simple_threshold(image_bgr, thresh_val=thresh_val)
    else:
        bin_img = binarize_otsu_inv(image_bgr)

    # 2) Clean & separate strokes
    bin_img = morph_close(bin_img, (3, 3), 1)
    bin_img = morph_open(bin_img, (2, 2), 1)

    # Optionally erode then dilate to break narrow bridges between digits
    if erode_iters and erode_iters > 0:
        bin_img = morph_erode(bin_img, ksize=erode_ksize, iters=erode_iters)
        bin_img = morph_dilate(bin_img, ksize=erode_ksize, iters=erode_iters)

    # 3) Try projection split first (line-level)
    boxes = split_digits_projection(bin_img, min_width=12, smooth_w=21, gap_rel=0.18, gap_min_w=2)

    # 4) If projection couldn't split well, try contour fallback
    if len(boxes) <= 1:
        boxes = split_digits_contours_fallback(bin_img, min_area=80, dilate_ksize=3, dilate_iters=1)

    # 5) Optional: watershed refinement for very wide boxes
    if use_watershed and len(boxes) > 0:
        refined = []
        for b in boxes:
            x1, y1, x2, y2 = b
            w, h = (x2 - x1), (y2 - y1)
            if w > 1.6 * h:  # more aggressive threshold for wide boxes
                parts = watershed_split(bin_img, b)
                refined.extend(parts)
            else:
                refined.append(b)
        boxes = sorted(refined, key=lambda r: r[0])

    # 6) Tighten boxes to remove whitespace (good for 7)
    boxes = tighten_boxes(bin_img, boxes, margin=2)

    # 7) Smart-split wide boxes using projection inside box (find valley)
    final = []
    for b in boxes:
        x1, y1, x2, y2 = b
        w, h = (x2 - x1), (y2 - y1)
        # heuristic: if box is considerably wider than expected, attempt smart split
        if w > 1.2 * h and w >= 18:
            parts = smart_split_box(bin_img, b, min_gap_width=3, smooth_w=9, gap_rel=0.18)
            # tighten returned parts as well
            parts = tighten_boxes(bin_img, parts, margin=2)
            final.extend(parts)
        else:
            final.append(b)

    boxes = sorted(final, key=lambda r: r[0])
    return boxes, bin_img

# --- utils ---

def save_crops(img, boxes, out_dir, resize_to=None):
    """Save cropped digit images into output directory. Optionally resize (w,h)."""
    os.makedirs(out_dir, exist_ok=True)
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        crop = img[y1:y2, x1:x2]
        if resize_to:
            crop = cv2.resize(crop, resize_to, interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(out_dir, f"digit_{i:02d}.png"), crop)

def draw_boxes(img, boxes, color=(0,255,0), thickness=2):
    """Draw green bounding boxes around digits."""
    vis = img.copy()
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
    return vis

# --- entry point with argparse ---

def parse_args():
    p = argparse.ArgumentParser(description="Digit segmentation from images (improved).")
    p.add_argument("image", help="Path to input image")
    p.add_argument("--no-watershed", action="store_true", help="Disable watershed refinement")
    p.add_argument("--simple-thresh", type=int, default=None, help="Use fixed global threshold with given value")
    p.add_argument("--adaptive", action="store_true", help="Use adaptive Gaussian thresholding")
    p.add_argument("--trim", action="store_true", help="Trim bounding boxes to digit edges (tighten)")
    p.add_argument("--no-contour-check", action="store_true", help="Disable contour refinement inside boxes")
    p.add_argument("--save", type=str, default=None, help="Directory to save cropped digit images")
    p.add_argument("--erode", type=int, default=0, help="Apply erosion+restore (iterations) to split touching digits (0=off)")
    p.add_argument("--resize", type=int, nargs=2, metavar=('W','H'), help="Resize saved crops to W H (e.g. 28 28)")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(args.image)

    boxes, bin_img = segment_digits(
        img,
        use_watershed=not args.no_watershed,
        use_simple_thresh=args.simple_thresh is not None,
        use_adaptive=args.adaptive,
        thresh_val=args.simple_thresh if args.simple_thresh else 128,
        erode_iters=args.erode,
        erode_ksize=(2,2)
    )

    # optional contour refinement inside each box (split multiple contours in a box)
    if not args.no_contour_check:
        refined = []
        for b in boxes:
            x1, y1, x2, y2 = b
            roi = bin_img[y1:y2, x1:x2]
            cnts, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(cnts) > 1:
                for c in cnts:
                    xx, yy, ww, hh = cv2.boundingRect(c)
                    refined.append((x1+xx, y1+yy, x1+xx+ww, y1+yy+hh))
            else:
                refined.append(b)
        boxes = sorted(refined, key=lambda r: r[0])

    # optionally tighten one more time if --trim used (keeps default behavior without --trim)
    if args.trim:
        boxes = tighten_boxes(bin_img, boxes, margin=2)

    # final sort
    boxes = sorted(boxes, key=lambda r: r[0])

    vis = draw_boxes(img, boxes)

    print(f"Found {len(boxes)} digit(s). Boxes (x1,y1,x2,y2):\n{boxes}")

    cv2.imshow("Binarized", bin_img)
    cv2.imshow("Digits (boxed)", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if args.save:
        resize_to = tuple(args.resize) if args.resize else None
        # Save crops in BGR (original) and in grayscale bin image for quick recognition
        save_crops(img, boxes, os.path.join(args.save, "color"), resize_to=resize_to)
        save_crops(cv2.cvtColor(bin_img, cv2.COLOR_GRAY2BGR), boxes, os.path.join(args.save, "bin"), resize_to=resize_to)
        print(f"Crops saved to: {args.save}")

    print("Running segmentation...")
    print("Arguments:", sys.argv)
