import sys
import os
import numpy as np
from tensorflow.keras.models import load_model

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.preprocessing import preprocess_image

# === CONFIG ===
MODEL_PATH = "models/cnn_model.keras"
IMG_SIZE = (128, 128)
CLASS_NAMES = ['Crazing', 'Inclusion', 'Patches', 'Pitted_Surface', 'Rolled_In_Scale', 'Scratches']

def predict_image(image_path):
    model = load_model(MODEL_PATH)
    img = preprocess_image(image_path, IMG_SIZE)
    img = np.expand_dims(img, axis=0)  # shape: (1, 128, 128, 3)
    prediction = model.predict(img)[0]
    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = np.max(prediction)
    return predicted_class, confidence

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 src/predict.py path/to/image.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    predicted_class, confidence = predict_image(image_path)
    print(f"Predicted class: {predicted_class} ({confidence*100:.2f}% confidence)")
