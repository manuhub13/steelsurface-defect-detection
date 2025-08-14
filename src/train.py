
import sys
import os
# Add the project root to the Python module search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from src.preprocessing import preprocess_image

# === CONFIG ===
DATA_DIR = 'data/NEU-CLS'
IMG_SIZE = (128, 128)
EPOCHS = 10
BATCH_SIZE = 32

# === Load and preprocess images ===
X = []
y = []
class_names = sorted(os.listdir(DATA_DIR))  # e.g. ['Crazing', 'Inclusion', ...]
class_to_idx = {name: i for i, name in enumerate(class_names)}

print(" Loading images...")

for class_name in class_names:
    folder = os.path.join(DATA_DIR, class_name)
    for filename in os.listdir(folder):
        if filename.endswith('.jpg'):
            image_path = os.path.join(folder, filename)
            img = preprocess_image(image_path, IMG_SIZE)
            X.append(img)
            y.append(class_to_idx[class_name])

X = np.array(X)
y = to_categorical(np.array(y))  # One-hot encoding

print(" Loaded", len(X), "images")

# === Split into training and validation ===
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# === Build CNN model ===
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(*IMG_SIZE, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(len(class_names), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print(" Training model...")
model.fit(X_train, y_train, validation_data=(X_val, y_val),
          epochs=EPOCHS, batch_size=BATCH_SIZE)

# === Save model ===
os.makedirs("models", exist_ok=True)
model.save("models/cnn_model.keras")
print("Model saved as models/cnn_model.keras")

