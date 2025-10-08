import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import sys

# ==========================================================
# CONFIGURATION
# ==========================================================
DATASET_PATH = "dataset/other_datasets"
IMG_HEIGHT, IMG_WIDTH = 64, 64
EPOCHS = 10
MODEL_PATH = "math_symbol_model.h5"

# ==========================================================
# 1. LOAD ALL IMAGES
# ==========================================================
def collect_images(root_folder):
    image_paths, labels = [], []
    for label_folder in os.listdir(root_folder):
        full_label_path = os.path.join(root_folder, label_folder)
        if not os.path.isdir(full_label_path):
            continue
        for file in os.listdir(full_label_path):
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                image_paths.append(os.path.join(full_label_path, file))
                labels.append(label_folder)
    return image_paths, labels

print("🔍 Scanning dataset...")
image_paths, labels = collect_images(DATASET_PATH)
print(f"✅ Found {len(image_paths)} images across {len(set(labels))} classes.")

# ==========================================================
# 2. ENCODE LABELS
# ==========================================================
label_names = sorted(list(set(labels)))
label_to_index = {name: i for i, name in enumerate(label_names)}
index_to_label = {i: name for name, i in label_to_index.items()}
print("Classes:", label_names)

y = np.array([label_to_index[label] for label in labels])

# ==========================================================
# 3. LOAD AND PREPROCESS IMAGES
# ==========================================================
def load_and_resize(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = img.astype("float32") / 255.0
    return img

X = np.array([load_and_resize(p) for p in image_paths])
print("📊 Dataset shape:", X.shape, y.shape)

# ==========================================================
# 4. TRAIN / TEST SPLIT
# ==========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

y_train = tf.keras.utils.to_categorical(y_train, len(label_names))
y_test = tf.keras.utils.to_categorical(y_test, len(label_names))

# ==========================================================
# 5. BUILD CNN MODEL
# ==========================================================
model = models.Sequential([
    layers.Conv2D(32, 3, activation="relu", input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, activation="relu"),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(len(label_names), activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# ==========================================================
# 6. TRAIN MODEL
# ==========================================================
print("🚀 Training model...")
history = model.fit(X_train, y_train, validation_split=0.1, epochs=EPOCHS, batch_size=32)
model.save(MODEL_PATH)
print(f"✅ Model saved as {MODEL_PATH}")

# ==========================================================
# 7. EVALUATE MODEL
# ==========================================================
loss, acc = model.evaluate(X_test, y_test)
print(f"📈 Test accuracy: {acc:.4f}")

# ==========================================================
# 8. SEGMENT AND SOLVE EQUATION
# ==========================================================
def segment_and_solve_equation(image_path):
    if not os.path.exists(image_path):
        print(f"❌ Error: File '{image_path}' not found")
        return None
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Error: Could not load image '{image_path}'")
        return None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    equation_parts = []
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        # Filter small noise
        if w > 10 and h > 10:
            symbol_img = img[y:y+h, x:x+w]
            symbol_img = cv2.resize(symbol_img, (IMG_WIDTH, IMG_HEIGHT))
            symbol_img = symbol_img.astype("float32") / 255.0
            
            # Predict symbol
            pred = model.predict(np.expand_dims(symbol_img, axis=0), verbose=0)
            symbol = index_to_label[np.argmax(pred)]
            confidence = np.max(pred)
            equation_parts.append((x, symbol, confidence))
    
    # Sort symbols by x-coordinate and build equation
    equation_parts.sort(key=lambda x: x[0])
    equation = ''.join([part[1] for part in equation_parts])
    
    # Show detected symbols and confidence
    print("\n🔍 Detected Symbols:")
    for i, (x, symbol, confidence) in enumerate(equation_parts):
        print(f"  Symbol {i+1}: '{symbol}' (confidence: {confidence:.4f})")
    
    print(f"\n🧮 Equation: {equation}")
    
    # Solve the equation
    try:
        # Replace operators for Python evaluation
        equation_eval = equation.replace('×', '*').replace('÷', '/')
        result = eval(equation_eval)
        print(f"✅ Answer: {result}")
        return result
    except Exception as e:
        print(f"❌ Could not solve equation: {str(e)}")
        return None

# ==========================================================
# 9. MAIN EXECUTION
# ==========================================================
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python train_math_solver.py <image_path>")
        print("Example: python train_math_solver.py equation.png")
        sys.exit(1)
    
    image_path = sys.argv[1]
    print(f"🎯 Processing image: {image_path}")
    segment_and_solve_equation(image_path)