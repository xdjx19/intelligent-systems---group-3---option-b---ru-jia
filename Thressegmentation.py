# Thressegmentation.py
# Stable 3-way splitter for joined digits:
# - Smoothed vertical projection to find two valleys
# - Corridor-aware recentering (low ink + far from strokes)
# - NEW: "left-guard" from tall-skinny component (the "1") to stop left cut drift
#
# pip install opencv-python numpy
#
# Run:
#   python Thressegmentation.py joint.png
#   python Thressegmentation.py joint.png --save out_digits

import cv2, numpy as np, os, sys

# ---------------- Binarization ----------------

def binarize_otsu_inv(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return th

def tight_bbox(bin_img):
    ys, xs = np.where(bin_img > 0)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1

# ---------------- Projection helpers ----------------

def smoothed_projection(roi, win):
    proj = (roi > 0).sum(axis=0).astype(np.float32)
    k = int(win); 
    if k % 2 == 0: k += 1
    k = max(3, k)
    kernel = np.ones(k, np.float32) / k
    return np.convolve(proj, kernel, mode="same"), proj

def pick_two_valleys(roi, margin=0.12, smooth_w=None, min_sep_frac=0.24, min_side_fg=0.008):
    """
    Two deep valleys, far from edges, well separated, both sides with ink.
    """
    H, W = roi.shape
    if W < 40:
        return []
    if smooth_w is None:
        smooth_w = max(7, (W // 18) | 1)

    smooth, _ = smoothed_projection(roi, smooth_w)
    mx = float(smooth.max()) if smooth.max() > 0 else 1.0

    L = int(W * margin)
    R = int(W * (1.0 - margin))
    if R <= L + 2:
        return []

    cand = []
    for x in range(L + 1, R - 1):
        if smooth[x] <= smooth[x - 1] and smooth[x] <= smooth[x + 1]:
            left_fg  = (roi[:, :x] > 0).sum()  / float(max(1, x * H))
            right_fg = (roi[:, x:] > 0).sum()  / float(max(1, (W - x) * H))
            if left_fg >= min_side_fg and right_fg >= min_side_fg:
                depth = 1.0 - (smooth[x] / mx)
                cand.append((x, depth))
    if not cand:
        j = L + int(np.argmin(smooth[L:R])) if R > L else W // 2
        return [j]

    cand.sort(key=lambda t: t[1], reverse=True)  # deepest first
    min_sep = int(W * min_sep_frac)
    picks = []
    for j, _ in cand:
        if not picks or all(abs(j - p) >= min_sep for p in picks):
            picks.append(j)
        if len(picks) == 2: break
    picks.sort()
    return picks

# ---------------- Corridor-aware recentering ----------------

def recenter_cut_to_local_min(roi, col, window=20, alpha=0.6):
    """
    Slide cut to best corridor within ±window:
      score = alpha * (normalized column ink)
            + (1 - alpha) * (1 - normalized distance)
    Low score => low ink + far from strokes.
    """
    H, W = roi.shape
    left  = max(0, col - window)
    right = min(W - 1, col + window)
    sub   = roi[:, left:right + 1]
    if sub.size == 0: 
        return col

    proj = (sub > 0).sum(axis=0).astype(np.float32)
    proj /= (proj.max() + 1e-6)

    dt = cv2.distanceTransform(255 - sub, cv2.DIST_L2, 5)
    dt_col = dt.mean(axis=0).astype(np.float32)
    dt_col /= (dt_col.max() + 1e-6)

    combined = alpha * proj + (1.0 - alpha) * (1.0 - dt_col)
    j_rel = int(np.argmin(combined))
    return int(left + j_rel)

# ---------------- NEW: left-guard from tall-skinny component ----------------

def estimate_left_guard(roi, ar_thresh=2.3, left_band=0.25, min_rel_h=0.55):
    """
    Try to find the '1': a tall, skinny component near the left.
    Returns x_right of that component (in ROI coords), or None.
    """
    H, W = roi.shape
    num, labels, stats, _ = cv2.connectedComponentsWithStats(roi, connectivity=8)
    best = None
    for i in range(1, num):  # skip background
        x, y, w, h, area = stats[i]
        if w <= 0 or h <= 0: 
            continue
        ar = h / (w + 1e-6)
        # near-left, tall, reasonably tall in absolute terms
        if (x < int(W * left_band)) and (ar >= ar_thresh) and (h >= min_rel_h * H):
            if best is None or x + w > best:  # rightmost edge among candidates
                best = x + w
    return best  # None or integer

# ---------------- Main segmentation (always 3 boxes) ----------------

def segment_three(img_bgr):
    bin_img = binarize_otsu_inv(img_bgr)

    tb = tight_bbox(bin_img)
    if tb is None:
        return [], bin_img
    x1, y1, x2, y2 = tb

    # small pad
    pad = 2
    x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
    x2 = min(bin_img.shape[1], x2 + pad); y2 = min(bin_img.shape[0], y2 + pad)

    roi = bin_img[y1:y2, x1:x2]
    H, W = roi.shape

    # 1) two valley candidates
    valleys = pick_two_valleys(roi, margin=0.12, smooth_w=None, min_sep_frac=0.24, min_side_fg=0.008)
    if len(valleys) < 2:
        valleys = [W // 3, 2 * W // 3]
    valleys = sorted(valleys[:2])

    # 2) corridor-aware recentering
    cuts = [recenter_cut_to_local_min(roi, c, window=20, alpha=0.6) for c in valleys]
    cuts.sort()

    # 3) guardrails
    min_width = max(14, W // 12)  # >= ~8% of width
    min_gap   = max(6,  W // 80)

    # --- NEW: left-guard using the tall-skinny '1' component ---
    left_guard = estimate_left_guard(roi, ar_thresh=2.3, left_band=0.25, min_rel_h=0.55)
    if left_guard is not None:
        # push first cut to the right of the '1' by a small margin
        guard_margin = max(6, int(0.02 * W))
        cuts[0] = max(cuts[0], left_guard + guard_margin)

    # keep inside and spaced
    valid = []
    for c in cuts:
        if c < min_width or (W - c) < min_width:
            continue
        if valid and abs(c - valid[-1]) < min_gap:
            continue
        valid.append(c)

    if len(valid) < 2:
        valid = [W // 3, 2 * W // 3]
    valid = valid[:2]
    c1, c2 = valid[0], valid[1]

    if c2 - c1 < min_gap:
        mid = (c1 + c2) // 2
        c1 = max(min_width, mid - min_gap // 2)
        c2 = min(W - min_width, mid + min_gap // 2)

    boxes = [
        (x1, y1, x1 + c1, y2),
        (x1 + c1, y1, x1 + c2, y2),
        (x1 + c2, y1, x2, y2)
    ]
    return boxes, bin_img

# ---------------- I/O & visualization ----------------

def draw_boxes(img, boxes):
    vis = img.copy()
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return vis

def save_crops(img, boxes, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        cv2.imwrite(os.path.join(out_dir, f"digit_{i:02d}.png"), img[y1:y2, x1:x2])

# ---------------- CLI ----------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python Thressegmentation.py <image_path> [--save out_dir]")
        sys.exit(1)

    img_path = sys.argv[1]
    out_dir = None
    if "--save" in sys.argv:
        i = sys.argv.index("--save")
        out_dir = sys.argv[i + 1] if i + 1 < len(sys.argv) else "out_digits"

    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(img_path)

    boxes, bin_img = segment_three(img)
    vis = draw_boxes(img, boxes)

    print(f"Found {len(boxes)} box(es): {boxes}")
    cv2.imshow("Binarized", bin_img)
    cv2.imshow("Digits", vis)
    cv2.waitKey(0)

    if out_dir:
        save_crops(cv2.cvtColor(bin_img, cv2.COLOR_GRAY2BGR), boxes, out_dir)
        print(f"Crops saved to: {out_dir}")
