# threshold_segmentation.py - CLEAN MNIST WITH OPERATOR RECOGNITION
import cv2
import numpy as np
import os
import sys
import argparse
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

# ==================== BINARIZATION ====================
def binarize_otsu_inv(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return th

# ==================== SEGMENTATION ====================
def segment_all_characters(bin_img):
    """Find ALL characters"""
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        if (w >= 3 and h >= 8 and w * h >= 20):
            boxes.append((x, y, x + w, y + h))
    
    boxes.sort(key=lambda b: b[0])
    return boxes

def tighten_boxes(bin_img, boxes, margin=1):
    H, W = bin_img.shape[:2]
    refined = []
    
    for box in boxes:
        x1, y1, x2, y2 = box
        roi = bin_img[y1:y2, x1:x2]
        ys, xs = np.where(roi > 0)
        
        if len(xs) == 0:
            refined.append(box)
            continue
            
        nx1 = max(0, x1 + xs.min() - margin)
        ny1 = max(0, y1 + ys.min() - margin)
        nx2 = min(W, x1 + xs.max() + margin)
        ny2 = min(H, y1 + ys.max() + margin)
        
        if (nx2 - nx1) >= 3 and (ny2 - ny1) >= 8:
            refined.append((nx1, ny1, nx2, ny2))
    
    return refined

# ==================== MNIST RECOGNIZER ====================
class MNISTRecognizer:
    def __init__(self):
        self.model = None
        self._train_model()
    
    def _train_model(self):
        """Train MNIST model without saving to disk"""
        dataset_folder = "dataset"
        train_path = os.path.join(dataset_folder, "mnist_train.csv")
        test_path = os.path.join(dataset_folder, "mnist_test.csv")
        
        if not os.path.exists(train_path):
            print(f"ERROR: Cannot find {train_path}")
            return
        
        print("Loading MNIST data...")
        train_data = pd.read_csv(train_path)
        
        # Prepare data
        X_train = train_data.iloc[:, 1:].values
        y_train = train_data.iloc[:, 0].values
        
        # Use test data if available, otherwise split
        if os.path.exists(test_path):
            test_data = pd.read_csv(test_path)
            X_test = test_data.iloc[:, 1:].values
            y_test = test_data.iloc[:, 0].values
        else:
            X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        
        # Normalize
        X_train = X_train / 255.0
        X_test = X_test / 255.0
        
        print(f"Training on {X_train.shape[0]} samples")
        
        # Train model
        self.model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=10, random_state=42)
        self.model.fit(X_train, y_train)
        
        accuracy = self.model.score(X_test, y_test)
        print(f"MNIST accuracy: {accuracy:.3f}")
    
    def preprocess_for_mnist(self, crop):
        if crop.size == 0:
            return None
        
        if len(crop.shape) == 3:
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        
        resized = cv2.resize(crop, (28, 28))
        
        if np.mean(resized) > 127:
            resized = 255 - resized
        
        normalized = resized.astype(np.float32) / 255.0
        flattened = normalized.reshape(1, -1)
        
        return flattened
    
    def recognize_digit(self, crop):
        if self.model is None:
            return None, 0
            
        preprocessed = self.preprocess_for_mnist(crop)
        if preprocessed is None:
            return None, 0
        
        try:
            prediction = self.model.predict(preprocessed)[0]
            confidence = np.max(self.model.predict_proba(preprocessed))
            return str(prediction), confidence
        except:
            return None, 0

# ==================== HYBRID RECOGNIZER ====================
class HybridRecognizer:
    def __init__(self):
        self.mnist = MNISTRecognizer()
    
    def is_operator(self, crop):
        """Check if character is likely an operator"""
        if crop.size == 0:
            return False, '?'
        
        h, w = crop.shape
        aspect_ratio = w / h
        fill_ratio = np.sum(crop > 0) / (w * h)
        
        # Operator detection rules
        if aspect_ratio > 2.0:
            horizontal_proj = np.sum(crop > 0, axis=1)
            peaks = len([i for i in range(1, len(horizontal_proj)-1) 
                        if horizontal_proj[i] > horizontal_proj[i-1] and 
                        horizontal_proj[i] > horizontal_proj[i+1]])
            return True, '=' if peaks >= 2 else '-'
        
        elif aspect_ratio < 0.3:
            return True, '|' if fill_ratio < 0.5 else '1'
        
        elif 0.8 < aspect_ratio < 1.2 and fill_ratio < 0.6:
            horizontal_proj = np.sum(crop > 0, axis=1)
            vertical_proj = np.sum(crop > 0, axis=0)
            center_h = horizontal_proj[h//2] > np.max(horizontal_proj) * 0.7
            center_v = vertical_proj[w//2] > np.max(vertical_proj) * 0.7
            if center_h and center_v:
                return True, '+'
        
        elif aspect_ratio < 0.6:
            if np.any(crop > 0):
                center_x = np.mean(np.where(crop > 0)[1]) / w
                return True, '(' if center_x < 0.4 else ')'
        
        elif 0.3 < aspect_ratio < 0.8 and fill_ratio < 0.3:
            return True, '/'
        
        elif fill_ratio < 0.4 and w > 15 and h > 15:
            return True, 'x' if aspect_ratio > 0.8 else 'y'
        
        return False, '?'
    
    def recognize_character(self, crop):
        is_op, operator = self.is_operator(crop)
        if is_op and operator != '?':
            return operator
        
        digit, confidence = self.mnist.recognize_digit(crop)
        
        if digit and confidence > 0.1:
            return digit
        elif operator != '?':
            return operator
        else:
            h, w = crop.shape
            fill_ratio = np.sum(crop > 0) / (w * h) if w * h > 0 else 0
            return '8' if fill_ratio > 0.6 else '1'
    
    def recognize_expression(self, bin_img, boxes):
        expression = []
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            crop = bin_img[y1:y2, x1:x2]
            
            symbol = self.recognize_character(crop)
            expression.append(symbol)
            
        return ''.join(expression)

# ==================== MAIN ====================
def segment_digits(image_bgr):
    bin_img = binarize_otsu_inv(image_bgr)
    boxes = segment_all_characters(bin_img)
    boxes = tighten_boxes(bin_img, boxes, margin=1)
    return boxes, bin_img

def draw_boxes(img, boxes, expression, color=(0,255,0), thickness=2):
    vis = img.copy()
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
        if i < len(expression):
            cv2.putText(vis, expression[i], (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return vis

def parse_args():
    p = argparse.ArgumentParser(description="Math Expression Recognition")
    p.add_argument("image", help="Path to input image")
    p.add_argument("--eval", action="store_true", help="Evaluate expression")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    img = cv2.imread(args.image)
    if img is None: 
        raise FileNotFoundError(args.image)

    print("Processing image...")
    
    boxes, bin_img = segment_digits(img)
    print(f"Found {len(boxes)} characters")
    
    recognizer = HybridRecognizer()
    expression = recognizer.recognize_expression(bin_img, boxes)
    
    print(f"Expression: {expression}")
    
    if args.eval:
        try:
            eval_expr = expression.replace('x', '2').replace('y', '3')
            result = eval(eval_expr)
            print(f"Result: {result}")
        except:
            print("Could not evaluate expression")
    
    result_img = draw_boxes(img, boxes, expression)
    cv2.imshow("Binary", bin_img)
    cv2.imshow("Recognition Result", result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()