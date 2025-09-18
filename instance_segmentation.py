# Usage:
#   python instance_segmentation.py input_image.png
#   python instance_segmentation.py input_image.png --fixed-thresh 150
#   python instance_segmentation.py input_image.png --skip-morph

import cv2
import numpy as np
import os
import sys
import random

# --- Image Preprocessing ---

def threshold_otsu_inverted(img_bgr):
    """Convert to grayscale and apply Otsu's thresholding with inversion."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return thresh

def threshold_fixed(img_bgr, thresh_val=128, invert=True):
    """Apply a fixed threshold to grayscale image."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    flag = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, thresh = cv2.threshold(gray, thresh_val, 255, flag)
    return thresh


# --- Morphological Operations ---

def morph_refine(bin_img, ksize=(3,3), close_iters=1, erode_iters=1):
    """Apply closing to fill gaps, then erosion to separate touching objects."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize)
    refined = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel, iterations=close_iters)
    refined = cv2.erode(refined, kernel, iterations=erode_iters)
    return refined


# --- Instance Segmentation Method ---

def segment_instances_watershed(bin_img, min_area=50, dist_threshold=0.4):
    """
    Segment digit instances using distance transform and watershed.
    - bin_img: binary image (255 for digits, 0 for background)
    - min_area: minimum pixel area for valid instances
    - dist_threshold: relative threshold for foreground detection (0-1)
    Returns: list of (mask, bbox) where mask is np.array, bbox is (x1,y1,x2,y2)
    """
    if bin_img.size == 0:
        return []

    dist_map = cv2.distanceTransform(bin_img, cv2.DIST_L2, 3)
    if dist_map.max() <= 0:
        return []

    _, sure_fg = cv2.threshold(dist_map, dist_threshold * dist_map.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)

    unknown = cv2.subtract(bin_img, sure_fg)

    num_labels, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    img_color = cv2.cvtColor(bin_img, cv2.COLOR_GRAY2BGR)
    cv2.watershed(img_color, markers)

    instances = []
    for label in range(2, num_labels + 1):
        mask = (markers == label).astype(np.uint8) * 255
        area = np.sum(mask > 0)
        if area < min_area:
            continue
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            continue
        x1, y1 = xs.min(), ys.min()
        x2, y2 = xs.max() + 1, ys.max() + 1
        instances.append((mask, (x1, y1, x2, y2)))

    instances.sort(key=lambda x: x[1][0])
    return instances


# --- Visualization and Output ---

def visualize_instances(img, instances):
    """Overlay colored masks and draw bounding boxes on the image."""
    vis = img.copy()
    colors = [(random.randint(50, 255), random.randint(50, 255), random.randint(50, 255)) 
              for _ in instances]
    for i, (mask, (x1, y1, x2, y2)) in enumerate(instances):
        color = colors[i]
        color_mask = np.zeros_like(vis)
        color_mask[mask > 0] = color
        vis = cv2.addWeighted(vis, 0.7, color_mask, 0.3, 0)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 1)
    return vis

def save_segmented_crops(img, instances, output_dir):
    """Save individual instance crops with masks applied."""
    os.makedirs(output_dir, exist_ok=True)
    for i, (mask, (x1, y1, x2, y2)) in enumerate(instances):
        crop = img[y1:y2, x1:x2].copy()
        mask_crop = mask[y1:y2, x1:x2]
        if crop.ndim == 3:
            mask_crop = cv2.cvtColor(mask_crop, cv2.COLOR_GRAY2BGR)
        crop = cv2.bitwise_and(crop, mask_crop)
        cv2.imwrite(os.path.join(output_dir, f"segment_{i:03d}.png"), crop)


# --- Main Segmentation Function ---

def process_digits(img_bgr, apply_morph=True, use_fixed_thresh=False, thresh_val=128, dist_threshold=0.4):
    """Segment digit instances from an input image."""
    if use_fixed_thresh:
        binary = threshold_fixed(img_bgr, thresh_val=thresh_val)
    else:
        binary = threshold_otsu_inverted(img_bgr)

    if apply_morph:
        binary = morph_refine(binary, ksize=(3, 3), close_iters=1, erode_iters=1)

    instances = segment_instances_watershed(binary, min_area=50, dist_threshold=dist_threshold)

    return instances, binary


# --- CLI Input ---

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python instance_segmentation.py <image_path> [--skip-morph] [--fixed-thresh val] [--dist-thresh val] [--save output_dir]")
        sys.exit(1)

    image_path = sys.argv[1]
    apply_morph = "--skip-morph" not in sys.argv

    use_fixed = "--fixed-thresh" in sys.argv
    thresh_val = 128
    if use_fixed:
        idx = sys.argv.index("--fixed-thresh")
        if idx + 1 < len(sys.argv):
            thresh_val = int(sys.argv[idx + 1])

    dist_threshold = 0.4
    if "--dist-thresh" in sys.argv:
        idx = sys.argv.index("--dist-thresh")
        if idx + 1 < len(sys.argv):
            dist_threshold = float(sys.argv[idx + 1])

    output_dir = None
    if "--save" in sys.argv:
        idx = sys.argv.index("--save")
        if idx + 1 < len(sys.argv):
            output_dir = sys.argv[idx + 1]

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    instances, binary = process_digits(img, apply_morph=apply_morph, 
                                      use_fixed_thresh=use_fixed, thresh_val=thresh_val,
                                      dist_threshold=dist_threshold)
    
    vis = visualize_instances(img, instances)
    
    num_instances = len(instances)
    bboxes = [inst[1] for inst in instances]
    print(f"Detected {num_instances} digit instance(s). Bounding boxes (x1,y1,x2,y2):\n{bboxes}")

    cv2.imshow("Binary Image", binary)
    cv2.imshow("Segmented Digits", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if output_dir:
        save_segmented_crops(img, instances, output_dir)
        print(f"Saved segmented crops to: {output_dir}")

print("Starting digit instance segmentation...")
print("Command-line args:", sys.argv)