import os
import sys
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# ==========================================================
# CONFIGURATION
# ==========================================================
DATASET_DIR = "dataset/other_datasets"
MODEL_PATH = "math_solver_model.keras"
IMG_SIZE = (64, 64)

# ==========================================================
# TRAINING SECTION
# ==========================================================
def load_dataset():
    images, labels = [], []
    classes = sorted(os.listdir(DATASET_DIR))

    for label in classes:
        folder = os.path.join(DATASET_DIR, label)
        if not os.path.isdir(folder):
            continue
        for file in os.listdir(folder):
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                img_path = os.path.join(folder, file)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                img = cv2.resize(img, IMG_SIZE)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = img / 255.0
                images.append(img)
                labels.append(classes.index(label))
    return np.array(images).reshape(-1, IMG_SIZE[0], IMG_SIZE[1], 1), np.array(labels), classes

def build_model(num_classes):
    # CNN model for digit recognition
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 1)),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax') # Softmax for multi-class classification
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_and_save():
    print("Loading dataset...")
    X, y, classes = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Loaded {len(X)} images across {len(classes)} classes.")

    model = build_model(len(classes))
    model.summary()

    print("\nTraining model...")
    model.fit(X_train, y_train, epochs=12, validation_data=(X_test, y_test))

    model.save(MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")
    return model, classes

# ==========================================================
# PREDICTION SECTION
# ==========================================================
operator_map = {"add":"+","sub":"-","mul":"*","div":"/","eq":"=","x":"*","y":"+","z":"-"}
digits_set = [str(i) for i in range(10)]
operators_set = ["add","sub","mul","div","x","y","z"] # Supported operators
equal_set = ["eq"]

def preprocess_digit_image(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape)==3 else img.copy()
    thresh = cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV,11,2)
    kernel = np.ones((2,2),np.uint8)
    thresh = cv2.dilate(thresh,kernel,iterations=1) # Morphological dilation

    h,w = thresh.shape
        # Resize and maintain ratio
    if w > h:
        new_w = IMG_SIZE[0]
        new_h = int(h*(new_w/w))
    else:
        new_h = IMG_SIZE[1]
        new_w = int(w*(new_h/h))
    resized = cv2.resize(thresh,(new_w,new_h))
    padded = np.zeros(IMG_SIZE,dtype=np.float32)
    padded[:new_h,:new_w] = resized/255.0
    padded = np.expand_dims(padded,axis=-1)
    return np.expand_dims(padded,axis=0)

def predict_with_heuristics(img_pre, model, classes, aspect_ratio, position):
    pred = model.predict(img_pre, verbose=0)[0]
    class_idx = np.argmax(pred)
    label = classes[class_idx]

    # --- Forced operator correction ---
    if position == 1:  # operator is always second
        if aspect_ratio < 1.2:
            label = "add"   # narrow → +
        elif aspect_ratio > 1.2:
            label = "mul"   # wide → *
    # Equal sign correction
    if aspect_ratio > 2.5:
        label = "eq"
    return label

def enforce_expression_pattern(predicted_labels):
    cleaned = []
    # first digit
    for l in predicted_labels:
        if l in digits_set:
            cleaned.append(l)
            predicted_labels.remove(l)
            break
    # operator
    for l in predicted_labels:
        if l in operators_set:
            cleaned.append(l)
            predicted_labels.remove(l)
            break
    # second digit
    for l in predicted_labels:
        if l in digits_set:
            cleaned.append(l)
            predicted_labels.remove(l)
            break
    # equal
    for l in predicted_labels:
        if l in equal_set:
            cleaned.append(l)
            break
    return cleaned

def predict_image_segments(image_path, model, classes):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Could not read {image_path}")
        return

    _, thresh = cv2.threshold(image,128,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    contours,_ = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = sorted([cv2.boundingRect(c) for c in contours],key=lambda b:b[0])

    predictions = []
    for i,(x,y,w,h) in enumerate(bounding_boxes):
        if w<10 or h<10:
            continue
        roi = image[y:y+h,x:x+w]
        img_pre = preprocess_digit_image(roi)
        aspect_ratio = w/float(h)
        label = predict_with_heuristics(img_pre, model, classes, aspect_ratio, position=i)
        print(f"Box {(x,y,w,h)} -> Predicted: {label}, Aspect ratio: {aspect_ratio:.2f}")
        predictions.append(label)

    # enforce pattern <digit> <operator> <digit> =
    cleaned = enforce_expression_pattern(predictions)
    expression_symbols = [operator_map.get(r,r) for r in cleaned]
    expression_str = " ".join(expression_symbols)
    print(f"\nDetected Expression: {expression_str}")

    expr_eval = expression_str.replace("=","").strip()
    if expr_eval:
        try:
            answer = eval(expr_eval)
            print(f"Computed Result: {answer}")
            return f"{expr_eval} = {answer}"
        except Exception as e:
            print(f"Could not compute result: {e}")
            return f"Could not compute: {expression_str}"
    else:
        print("No valid expression detected.")
        return "No valid expression detected."

# ==========================================================
# MAIN EXECUTION
# ==========================================================
if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        model, classes = train_and_save()
    else:
        print(f"Loading existing model from {MODEL_PATH}")
        model = tf.keras.models.load_model(MODEL_PATH)
        _, _, classes = load_dataset()

    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        predict_image_segments(image_path, model, classes)
    else:
        print("No image provided. Run as:")
        print("   python train_math_solver.py eq.png")
