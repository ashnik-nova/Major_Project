# routes/dogcat_h5.py
from flask import Blueprint, request, jsonify
import numpy as np
from PIL import Image
import io
import os
from tensorflow import keras

dogcat_bp = Blueprint("dogcat", __name__)

# ---------- CONFIG: point to your .h5 file ----------
DOGCAT_H5_PATH = os.path.join("models", "dog_cat", "dogcat.h5")
# If your .h5 is somewhere else, update DOGCAT_H5_PATH accordingly.
# Example from your environment: B:\CodingStuffs\XAI_Project\Backend\models\dogcat\reconstructed.h5

# Load Keras .h5 model (cached)
if not os.path.exists(DOGCAT_H5_PATH):
    raise FileNotFoundError(f"Dog/Cat .h5 model not found at: {DOGCAT_H5_PATH}")

dogcat_model = keras.models.load_model(DOGCAT_H5_PATH, compile=False)

# Model expects 128x128
IMG_SIZE = (128, 128)

def preprocess_image(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = image.resize(IMG_SIZE)
        arr = np.array(image).astype("float32") / 255.0  # normalize as during training
        arr = np.expand_dims(arr, axis=0)  # shape (1, H, W, 3)
        return arr
    except Exception:
        return None

@dogcat_bp.route("/", methods=["POST"])
def predict_dogcat():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded", "tip": "Use form-data key 'image'"}), 400

    file_bytes = request.files["image"].read()
    img = preprocess_image(file_bytes)

    if img is None:
        return jsonify({"error": "Invalid image format"}), 400

    # Keras model inference
    preds = dogcat_model.predict(img)  # shape (1,1) for binary sigmoid
    # handle cases where model returns (1,) or (1,1)
    pred_val = float(np.asarray(preds).reshape(-1)[0])

    # Determine label and confidence
    if pred_val >= 0.5:
        label = "Dog"
        confidence = pred_val
    else:
        label = "Cat"
        confidence = 1.0 - pred_val

    return jsonify({
        "model": "Dog vs Cat Classifier (.h5)",
        "prediction": label,
        "confidence": float(round(confidence, 4)),
        "raw_score": float(round(pred_val, 6)),
        "test_image_path": "/mnt/data/5bee092b-485d-42c4-8edf-875c2af77927.png"
    }), 200
