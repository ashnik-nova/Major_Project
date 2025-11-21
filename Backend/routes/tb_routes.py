# routes/tb_h5.py
from flask import Blueprint, request, jsonify
import numpy as np
from PIL import Image
import io
import os
from tensorflow import keras

tb_bp = Blueprint("tb", __name__)

# ------------ CONFIG: .h5 MODEL PATH ------------
TB_H5_PATH = os.path.join("models", "tb", "tb.h5")
# Update if your file is somewhere else:
# B:\CodingStuffs\XAI_Project\Backend\models\tb\reconstructed.h5

if not os.path.exists(TB_H5_PATH):
    raise FileNotFoundError(f"TB model .h5 not found at: {TB_H5_PATH}")

# Load Keras .h5 model
tb_model = keras.models.load_model(TB_H5_PATH, compile=False)

# TB model input size (from your architecture)
IMG_SIZE = (224, 224)


def preprocess_image(image_bytes):
    """Load + resize + normalize image."""
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize(IMG_SIZE)
        arr = np.array(img).astype("float32") / 255.0
        arr = np.expand_dims(arr, axis=0)  # (1,H,W,3)
        return arr
    except Exception as e:
        print("Error preprocessing image:", e)
        return None


@tb_bp.route("/", methods=["POST"])
def predict_tb():
    """Predict TB vs Normal using .h5 model."""
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_bytes = request.files["image"].read()
    img = preprocess_image(image_bytes)

    if img is None:
        return jsonify({"error": "Invalid or corrupted image"}), 400

    # Run prediction
    preds = tb_model.predict(img)
    pred_val = float(np.asarray(preds).reshape(-1)[0])  # sigmoid output

    # Decide final class
    if pred_val >= 0.5:
        label = "TB Detected"
        confidence = pred_val
    else:
        label = "Normal"
        confidence = 1 - pred_val

    return jsonify({
        "model": "TB Classifier (.h5)",
        "prediction": label,
        "confidence": float(round(confidence, 4)),
        "raw_score": float(round(pred_val, 6))
    }), 200
