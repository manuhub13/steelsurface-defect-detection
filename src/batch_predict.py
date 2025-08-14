import os
import sys

# Get project root (one level above "src")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Debug check
print("Python search paths:", sys.path)
print("Looking for:", os.path.join(ROOT, "src", "preprocessing.py"))





import os
import cv2
import csv
import numpy as np
import tensorflow as tf
from src.preprocessing import preprocess_image

# Paths
MODEL_PATH = "models/cnn_model.keras"
DATA_DIR = "data/NEU-CLS"
OUTPUT_CSV = "outputs/predictions.csv"
LABELED_DIR = "outputs/labeled_images"

# Make sure output folders exist
os.makedirs(LABELED_DIR, exist_ok=True)

def draw_prediction_overlay_bounded(bgr_img, label_text, conf, correct, target_w=512):
    """
    - Upscales image to ~target_w for readability
    - Draws a boxed label that always fits (1 or 2 lines)
    - Returns the annotated (possibly upscaled) image
    """
    import cv2
    import numpy as np

    # --- upscale first so small images have more room for text ---
    h, w = bgr_img.shape[:2]
    scale = max(1, int(round(target_w / max(1, w))))
    vis = cv2.resize(bgr_img, (w * scale, h * scale), interpolation=cv2.INTER_LINEAR)

    h, w = vis.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (0, 200, 0) if correct else (0, 0, 255)
    full_text = f"Predicted: {label_text} ({conf*100:.1f}%)"

    # start from a sensible font size relative to width; we will shrink as needed
    font_scale = max(0.3, min(1.2, w / 500.0))
    thickness = max(1, int(round(font_scale * 2)))
    pad = int(round(6 * font_scale))

    # measure single-line
    (tw, th), base = cv2.getTextSize(full_text, font, font_scale, thickness)

    # shrink font until it fits OR switch to two-line
    tries = 0
    while tw + 2 * pad > w and font_scale > 0.3 and tries < 20:
        font_scale *= 0.9
        thickness = max(1, int(round(font_scale * 2)))
        pad = int(round(6 * font_scale))
        (tw, th), base = cv2.getTextSize(full_text, font, font_scale, thickness)
        tries += 1

    def boxed_text(img, text, x, y):
        (t_w, t_h), b = cv2.getTextSize(text, font, font_scale, thickness)
        cv2.rectangle(img, (x - pad, y - t_h - pad), (x + t_w + pad, y + b + pad), (0, 0, 0), -1)
        cv2.putText(img, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
        return t_h, b

    x = pad
    y = pad  # top-left anchor

    # If it fits in one line, draw once; else split into two lines.
    if tw + 2 * pad <= w:
        y_text = y + th + pad
        boxed_text(vis, full_text, x, y_text)
    else:
        line1 = f"Predicted: {label_text}"
        line2 = f"({conf*100:.1f}%)"
        (w1, h1), b1 = cv2.getTextSize(line1, font, font_scale, thickness)
        (w2, h2), b2 = cv2.getTextSize(line2, font, font_scale, thickness)

        y1 = y + h1 + pad
        boxed_text(vis, line1, x, y1)

        y2 = y1 + h2 + pad * 2
        boxed_text(vis, line2, x, y2)

    return vis


# Load model
model = tf.keras.models.load_model(MODEL_PATH)
class_names = sorted(os.listdir(DATA_DIR))

# Open CSV for writing
with open(OUTPUT_CSV, mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["Filename", "TrueLabel", "PredictedLabel", "Confidence"])

    # Loop through dataset
    for class_name in class_names:
        class_path = os.path.join(DATA_DIR, class_name)
        for filename in os.listdir(class_path):
            image_path = os.path.join(class_path, filename)

            # Preprocess
            img = preprocess_image(image_path)
            img_batch = np.expand_dims(img, axis=0)

            # Predict
            preds = model.predict(img_batch, verbose=0)
            pred_idx = np.argmax(preds, axis=1)[0]
            pred_label = class_names[pred_idx]
            conf = float(np.max(preds))

            # Write to CSV
            writer.writerow([filename, class_name, pred_label, f"{conf:.4f}"])

            # --- OpenCV overlay ---
            orig = cv2.imread(image_path)  # BGR
            h, w = orig.shape[:2]

            full_text = f"Predicted: {pred_label} ({conf*100:.1f}%)"
            font = cv2.FONT_HERSHEY_SIMPLEX

            # Start with a reasonable scale; shrink until it fits
            font_scale = w / 400.0
            font_scale = max(0.3, min(1.2, font_scale))
            thickness = max(1, int(round(font_scale * 2)))
            pad = int(round(6 * font_scale))

            # Measure single-line width
            (text_w, text_h), baseline = cv2.getTextSize(full_text, font, font_scale, thickness)

            # Shrink until fits
            tries = 0
            while text_w + 2 * pad > w and font_scale > 0.3 and tries < 20:
                font_scale *= 0.9
                thickness = max(1, int(round(font_scale * 2)))
                pad = int(round(6 * font_scale))
                (text_w, text_h), baseline = cv2.getTextSize(full_text, font, font_scale, thickness)
                tries += 1

            def draw_boxed_text(img, text, x, y, font, fs, thick, pad, color):
                (tw, th), base = cv2.getTextSize(text, font, fs, thick)
                cv2.rectangle(img,
                              (x - pad, y - th - pad),
                              (x + tw + pad, y + base + pad),
                              (0, 0, 0), thickness=-1)
                cv2.putText(img, text, (x, y), font, fs, color, thick, cv2.LINE_AA)

            # Color green if correct, red if wrong
            color = (0, 200, 0) if pred_label == class_name else (0, 0, 255)
            x = pad
            y = pad  # top-left origin

            if text_w + 2 * pad <= w:
                # Single-line fits
                y_text = y + text_h + pad
                draw_boxed_text(orig, full_text, x, y_text, font, font_scale, thickness, pad, color)
            else:
                # Two lines
                line1 = f"Predicted: {pred_label}"
                line2 = f"({conf*100:.1f}%)"
                (w1, h1), b1 = cv2.getTextSize(line1, font, font_scale, thickness)
                (w2, h2), b2 = cv2.getTextSize(line2, font, font_scale, thickness)
                y_line1 = y + h1 + pad
                draw_boxed_text(orig, line1, x, y_line1, font, font_scale, thickness, pad, color)
                y_line2 = y_line1 + h2 + pad*2
                draw_boxed_text(orig, line2, x, y_line2, font, font_scale, thickness, pad, color)

            # Optional upscale for easier viewing
            target_w = 512
            scale = max(1, int(round(target_w / w)))
            vis = cv2.resize(orig, (w * scale, h * scale), interpolation=cv2.INTER_LINEAR)

            # Save labeled image
            out_name = f"{class_name}_{os.path.splitext(filename)[0]}_v2.jpg"
            cv2.imwrite(os.path.join(LABELED_DIR, out_name), vis)

print(f"✅ Predictions complete. CSV saved to {OUTPUT_CSV}")
print(f"✅ Labeled images saved to {LABELED_DIR}")

