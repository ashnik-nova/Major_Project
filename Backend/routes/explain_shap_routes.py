# routes/gradcam_shap_highres.py
import os
import time
import traceback
import base64
import threading
from datetime import datetime, timedelta

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from flask import Blueprint, request, jsonify

# Optional: SHAP import
try:
    import shap
except Exception:
    shap = None

gradcam_shap_bp = Blueprint("gradcam_shap_bp", __name__)

# ------------- CONFIG -------------
MODEL_FILES = {
    "dog_cat": {
        "h5": r'B:\CodingStuffs\XAI_Project\Backend\models\dog_cat\dogcat.h5',
        "img_size": (128, 128),
        "last_conv": "conv2d_4",
        "class_names": ["Cat", "Dog"]
    },
    "tb": {
        "h5": r'B:\CodingStuffs\XAI_Project\Backend\models\tb\tb.h5',
        "img_size": (224, 224),
        "last_conv": "conv2d_5",
        "class_names": ["Normal", "Tuberculosis"]
    }
}

FEATUREMAP_DIR = os.path.join("static", "featuremaps")
GRADCAM_DIR = os.path.join(FEATUREMAP_DIR, "gradcam")
SHAP_DIR = os.path.join(FEATUREMAP_DIR, "shap")
UPLOAD_DIR = "uploads"
os.makedirs(GRADCAM_DIR, exist_ok=True)
os.makedirs(SHAP_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# caches
_LOADED_MODELS = {}
_SHAP_EXPLAINERS = {}

# ------------- AUTO-CLEANUP SYSTEM -------------
_CLEANUP_REGISTRY = []  # List of (filepath, deletion_time)
_CLEANUP_LOCK = threading.Lock()

def schedule_file_deletion(filepath, delay_seconds=60):
    """
    Schedule a file for deletion after delay_seconds.
    """
    deletion_time = datetime.now() + timedelta(seconds=delay_seconds)
    with _CLEANUP_LOCK:
        _CLEANUP_REGISTRY.append((filepath, deletion_time))
    print(f"[SCHEDULED] {filepath} will be deleted at {deletion_time.strftime('%H:%M:%S')}")

def cleanup_worker():
    """
    Background thread that periodically checks and deletes expired files.
    Runs every 10 seconds.
    """
    print("[CLEANUP WORKER] Started")
    while True:
        time.sleep(10)  # Check every 10 seconds
        now = datetime.now()
        with _CLEANUP_LOCK:
            to_delete = []
            remaining = []
            
            for filepath, deletion_time in _CLEANUP_REGISTRY:
                if now >= deletion_time:
                    to_delete.append(filepath)
                else:
                    remaining.append((filepath, deletion_time))
            
            _CLEANUP_REGISTRY[:] = remaining
        
        # Delete files outside the lock to avoid blocking
        for filepath in to_delete:
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
                    print(f"[CLEANUP] ✓ Deleted: {filepath}")
            except Exception as e:
                print(f"[CLEANUP ERROR] ✗ Failed to delete {filepath}: {e}")

# Start cleanup thread when module loads
_cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
_cleanup_thread.start()

# ------------- UTIL -------------
def load_model_h5(key):
    if key in _LOADED_MODELS:
        return _LOADED_MODELS[key]
    cfg = MODEL_FILES.get(key)
    if not cfg:
        raise ValueError(f"Unknown model key: {key}")
    path = cfg["h5"]
    if not os.path.exists(path):
        raise FileNotFoundError(f".h5 model not found: {path}")
    model = keras.models.load_model(path, compile=False)
    _LOADED_MODELS[key] = model
    return model

def save_rgb_image_and_url(rgb_img, out_dir, prefix="out", auto_delete=True, delete_after=60):
    """
    Save RGB image to disk and return URL path.
    
    Args:
        rgb_img: RGB image array
        out_dir: Output directory
        prefix: Filename prefix
        auto_delete: Whether to automatically delete after delay
        delete_after: Seconds after which to delete (default: 60)
    """
    ts = int(time.time() * 1000)
    fname = f"{prefix}_{ts}.png"
    disk = os.path.join(out_dir, fname)
    # ensure out_dir exists
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(disk, cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))
    rel = f"/static/featuremaps/{os.path.basename(out_dir)}/{fname}"
    
    # Schedule for automatic deletion
    if auto_delete:
        schedule_file_deletion(disk, delay_seconds=delete_after)
    
    return rel, disk

def img_to_base64(rgb_img):
    _, buf = cv2.imencode('.png', cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode('ascii')

def preprocess_image_bytes(image_bytes, target_size):
    """
    returns (1,H,W,3) float32 in [0,1], original_bgr (uint8), and resized_bgr (uint8)
    """
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if bgr is None:
            return None, None, None
        orig_bgr = bgr.copy()
        # ensure target_size is (w,h) for cv2 resizing
        resized = cv2.resize(bgr, target_size, interpolation=cv2.INTER_AREA)
        resized_bgr = resized.copy()
        resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        x = resized_rgb.astype("float32") / 255.0
        return np.expand_dims(x, axis=0), orig_bgr, resized_bgr
    except Exception:
        return None, None, None

def apply_heatmap_overlay(heatmap, base_image_bgr, alpha=0.6, colormap=cv2.COLORMAP_JET, 
                         blur_ksize=(15,15)):
    """
    Apply heatmap overlay - heatmap is resized to match base_image dimensions
    """
    bh, bw = base_image_bgr.shape[:2]
    heat = cv2.resize(heatmap, (bw, bh), interpolation=cv2.INTER_LANCZOS4)
    heat = np.clip(heat, 0, 1)
    try:
        heat = cv2.bilateralFilter(heat.astype(np.float32), d=9, sigmaColor=0.1, sigmaSpace=9)
    except Exception:
        pass
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
    if blur_ksize:
        kx, ky = blur_ksize
        kx = kx if kx % 2 == 1 else kx + 1
        ky = ky if ky % 2 == 1 else ky + 1
        heat = cv2.GaussianBlur(heat, (kx, ky), sigmaX=0, sigmaY=0)
    heat_u8 = np.uint8(255 * heat)
    heat_bgr = cv2.applyColorMap(heat_u8, colormap)
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)
    base_rgb = cv2.cvtColor(base_image_bgr, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(base_rgb, alpha, heat_rgb, (1.0 - alpha), 0)
    return heat_rgb, overlay

# ------------- Grad-CAM -------------
def compute_gradcam_highres(model, last_conv_name, preproc_batch, orig_image_bgr,
                            blur_ksize=(15,15), alpha=0.6, colormap=cv2.COLORMAP_JET):
    """
    Compute Grad-CAM with overlay on original image size
    """
    try:
        last_conv = model.get_layer(last_conv_name)
    except Exception as e:
        raise ValueError(f"Last conv layer not found: {e}")

    grad_model = keras.models.Model(inputs=model.input, outputs=[last_conv.output, model.output])

    x = tf.convert_to_tensor(preproc_batch)
    with tf.GradientTape() as tape:
        tape.watch(x)
        conv_outputs, preds = grad_model(x, training=False)
        class_output = preds[:, 0]
    grads = tape.gradient(class_output, conv_outputs)
    if grads is None:
        raise RuntimeError("Could not compute gradients (None).")

    pooled = tf.reduce_mean(grads, axis=(0, 1, 2)).numpy()
    conv_np = conv_outputs[0].numpy()

    for i in range(pooled.shape[-1]):
        conv_np[:, :, i] *= pooled[i]

    cam = np.sum(conv_np, axis=-1)
    cam = np.maximum(cam, 0)

    if cam.max() != 0:
        cam = cam / (cam.max() + 1e-8)

    heat_rgb, overlay_rgb = apply_heatmap_overlay(
        heatmap=cam,
        base_image_bgr=orig_image_bgr,
        alpha=alpha,
        colormap=colormap,
        blur_ksize=blur_ksize
    )

    pred_score = float(preds.numpy()[0, 0])
    pred_class = int(np.round(pred_score))

    return pred_score, pred_class, heat_rgb, overlay_rgb

# ------------- SHAP (GradientExplainer) -------------
def get_or_create_shap_gradient_explainer(model_key, model, img_size):
    """
    Create SHAP GradientExplainer with a small diverse background.
    """
    if shap is None:
        raise RuntimeError("shap package not available.")
    if model_key in _SHAP_EXPLAINERS:
        return _SHAP_EXPLAINERS[model_key]
    background = np.zeros((5, img_size[0], img_size[1], 3), dtype='float32')
    for i in range(5):
        background[i] = np.full((img_size[0], img_size[1], 3), fill_value=(i / 4.0), dtype='float32')
    explainer = shap.GradientExplainer(model, background)
    _SHAP_EXPLAINERS[model_key] = explainer
    return explainer

def compute_shap_overlay_gradient(explainer, input_batch, orig_bgr, 
                                 blur_ksize=(15,15), alpha=0.6):
    """
    Compute SHAP gradient explanation with overlay on original image.
    """
    shap_values = explainer.shap_values(input_batch)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    vals = shap_values[0]
    saliency = np.sum(np.abs(vals), axis=-1)
    saliency = cv2.GaussianBlur(saliency, (5, 5), 0)

    thresh = np.percentile(saliency, 80)
    saliency = np.maximum(saliency - thresh, 0.0)

    if saliency.max() > 0:
        saliency = saliency / (saliency.max() + 1e-8)
    else:
        thresh = np.percentile(np.sum(np.abs(vals), axis=-1), 60)
        saliency = np.maximum(np.sum(np.abs(vals), axis=-1) - thresh, 0.0)
        if saliency.max() > 0:
            saliency = saliency / (saliency.max() + 1e-8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    sal_u8 = np.uint8(saliency * 255)
    sal_u8 = cv2.morphologyEx(sal_u8, cv2.MORPH_OPEN, kernel)
    saliency = sal_u8.astype('float32') / 255.0

    saliency = cv2.GaussianBlur(saliency, (7, 7), 0)
    try:
        saliency = cv2.bilateralFilter(saliency.astype(np.float32), d=9, sigmaColor=0.1, sigmaSpace=9)
    except Exception:
        pass

    saliency = np.power(saliency, 2.0)

    if saliency.max() > 0:
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)

    heat_rgb, overlay_rgb = apply_heatmap_overlay(
        heatmap=saliency,
        base_image_bgr=orig_bgr,
        alpha=alpha,
        colormap=cv2.COLORMAP_JET,
        blur_ksize=blur_ksize
    )

    return shap_values, heat_rgb, overlay_rgb

# ------------- Endpoints -------------
@gradcam_shap_bp.route("/gradcam", methods=["POST"])
def endpoint_gradcam():
    tmp = None
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image provided (use key 'image')"}), 400

        model_key = request.form.get('model')
        if not model_key or model_key not in MODEL_FILES:
            return jsonify({"error": f"Invalid model. Choose {list(MODEL_FILES.keys())}"}), 400

        blur_k = int(request.form.get('blur', 15))
        alpha = float(request.form.get('alpha', 0.6))

        upload = request.files['image']
        fname = upload.filename or f"upload_{int(time.time()*1000)}.png"
        tmp = os.path.join(UPLOAD_DIR, fname)
        upload.save(tmp)

        batch, orig_bgr, resized_bgr = preprocess_image_bytes(
            open(tmp, 'rb').read(), 
            MODEL_FILES[model_key]['img_size']
        )
        if batch is None or orig_bgr is None:
            raise ValueError("Uploaded image unreadable or corrupted.")

        model = load_model_h5(model_key)
        pred_score, pred_class, heat_rgb, overlay_rgb = compute_gradcam_highres(
            model=model,
            last_conv_name=MODEL_FILES[model_key]['last_conv'],
            preproc_batch=batch,
            orig_image_bgr=orig_bgr,
            blur_ksize=(blur_k, blur_k),
            alpha=alpha
        )

        # Save with auto-delete enabled (default 60 seconds)
        url, disk = save_rgb_image_and_url(overlay_rgb, GRADCAM_DIR, prefix=f"{model_key}_gradcam")
        b64 = img_to_base64(overlay_rgb)

        # Delete uploaded temp file immediately
        try:
            os.remove(tmp)
        except:
            pass

        return jsonify({
            "status": "success",
            "model": model_key,
            "prediction_index": int(pred_class),
            "prediction_label": MODEL_FILES[model_key]["class_names"][pred_class],
            "score": float(pred_score),
            "overlay_url": url,
            "overlay_base64": b64,
            "output_size": overlay_rgb.shape[:2],
            "auto_delete_info": "This image will be automatically deleted after 60 seconds"
        }), 200

    except Exception as e:
        traceback.print_exc()
        try:
            if tmp and os.path.exists(tmp): os.remove(tmp)
        except: pass
        return jsonify({"error": str(e)}), 500

@gradcam_shap_bp.route("/shap", methods=["POST"])
def endpoint_shap():
    tmp = None
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image provided (use key 'image')"}), 400

        if shap is None:
            return jsonify({"error": "shap package not installed."}), 500

        model_key = request.form.get('model')
        if not model_key or model_key not in MODEL_FILES:
            return jsonify({"error": f"Invalid model. Choose {list(MODEL_FILES.keys())}"}), 400

        blur_k = int(request.form.get('blur', 15))
        alpha = float(request.form.get('alpha', 0.6))
        shap_max = int(request.form.get('shap_max_evals', 80))

        upload = request.files['image']
        fname = upload.filename or f"upload_{int(time.time()*1000)}.png"
        tmp = os.path.join(UPLOAD_DIR, fname)
        upload.save(tmp)

        batch, orig_bgr, resized_bgr = preprocess_image_bytes(
            open(tmp, 'rb').read(), 
            MODEL_FILES[model_key]['img_size']
        )
        if batch is None:
            raise ValueError("Uploaded image unreadable or corrupted.")

        model = load_model_h5(model_key)

        preds = model.predict(batch)
        raw = float(np.asarray(preds).reshape(-1)[0])
        pred_class = int(np.round(raw))

        explainer = get_or_create_shap_gradient_explainer(
            model_key, model, MODEL_FILES[model_key]['img_size']
        )

        shap_vals, heat_rgb, overlay_rgb = compute_shap_overlay_gradient(
            explainer=explainer,
            input_batch=batch,
            orig_bgr=orig_bgr,
            blur_ksize=(blur_k, blur_k),
            alpha=alpha
        )

        # Save with auto-delete enabled (default 60 seconds)
        url, disk = save_rgb_image_and_url(overlay_rgb, SHAP_DIR, prefix=f"{model_key}_shap")
        b64 = img_to_base64(overlay_rgb)

        # Delete uploaded temp file immediately
        try:
            os.remove(tmp)
        except:
            pass

        return jsonify({
            "status": "success",
            "model": model_key,
            "prediction_index": int(pred_class),
            "prediction_label": MODEL_FILES[model_key]["class_names"][pred_class],
            "score": float(raw),
            "shap_overlay_url": url,
            "shap_overlay_base64": b64,
            "output_size": overlay_rgb.shape[:2],
            "shap_info": {
                "explainer_type": "GradientExplainer",
                "description": "Uses backpropagation for fast CNN explanations",
                "shap_max_evals_used": shap_max
            },
            "auto_delete_info": "This image will be automatically deleted after 60 seconds"
        }), 200

    except Exception as e:
        traceback.print_exc()
        try:
            if tmp and os.path.exists(tmp): os.remove(tmp)
        except: pass
        return jsonify({"error": str(e)}), 500