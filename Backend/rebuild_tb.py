# rebuild_tb_full.py
"""
Full script to:
 - rebuild the TB model with exact layer names matching your HDF5 weights
 - load weights from the HDF5 structure layers/<name>/vars/<i>
 - verify gradients and compute Grad-CAM (last conv: conv2d_5)
 - save reconstructed model to .keras and .h5
 - save Grad-CAM overlay image
Run inside your TF 2.15 virtualenv.
"""

import os
import h5py
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import cv2

# ========== CONFIG - update if needed ==========
WEIGHTS_H5 = r'B:\CodingStuffs\XAI_Project\Backend\models\tb\json_weights\model.weights.h5'
TEST_IMAGE = r'B:\CodingStuffs\XAI_Project\Backend\static\Tuberculosis-147.png'   # put a sample image here (optional)
OUT_DIR = r'B:\CodingStuffs\XAI_Project\Backend\gradcam_output_new'
SAVE_PATH_KERAS = r'B:\CodingStuffs\XAI_Project\Backend\models\tb\reconstructed.keras'
SAVE_PATH_H5 = r'B:\CodingStuffs\XAI_Project\Backend\models\tb\reconstructed.h5'
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(SAVE_PATH_H5), exist_ok=True)
# ===============================================

def build_model_matching_h5(input_shape=(224,224,3)):
    """Build model with layer names matching the HDF5 naming discovered earlier."""
    m = keras.Sequential(name='sequential_4')

    # block 1
    m.add(layers.Conv2D(32, (3,3), activation='relu', padding='same',
                        input_shape=input_shape, name='conv2d'))
    m.add(layers.BatchNormalization(name='batch_normalization'))
    m.add(layers.Conv2D(32, (3,3), activation='relu', padding='same',
                        name='conv2d_1'))
    m.add(layers.BatchNormalization(name='batch_normalization_1'))
    m.add(layers.MaxPooling2D((2,2), name='max_pooling2d'))
    m.add(layers.Dropout(0.0, name='dropout'))

    # block 2
    m.add(layers.Conv2D(64, (3,3), activation='relu', padding='same', name='conv2d_2'))
    m.add(layers.BatchNormalization(name='batch_normalization_2'))
    m.add(layers.Conv2D(64, (3,3), activation='relu', padding='same', name='conv2d_3'))
    m.add(layers.BatchNormalization(name='batch_normalization_3'))
    m.add(layers.MaxPooling2D((2,2), name='max_pooling2d_1'))
    m.add(layers.Dropout(0.0, name='dropout_1'))

    # block 3
    m.add(layers.Conv2D(128, (3,3), activation='relu', padding='same', name='conv2d_4'))
    m.add(layers.BatchNormalization(name='batch_normalization_4'))
    m.add(layers.Conv2D(128, (3,3), activation='relu', padding='same', name='conv2d_5'))
    m.add(layers.BatchNormalization(name='batch_normalization_5'))
    m.add(layers.MaxPooling2D((2,2), name='max_pooling2d_2'))
    m.add(layers.Dropout(0.0, name='dropout_2'))

    # dense head
    m.add(layers.Flatten(name='flatten'))
    m.add(layers.Dense(256, activation='relu', name='dense'))
    m.add(layers.BatchNormalization(name='batch_normalization_6'))
    m.add(layers.Dropout(0.0, name='dropout_3'))
    m.add(layers.Dense(128, activation='relu', name='dense_1'))
    m.add(layers.Dropout(0.0, name='dropout_4'))
    m.add(layers.Dense(1, activation='sigmoid', name='dense_2'))

    return m

def load_weights_layers_vars(model, h5_path):
    """
    Load weights from HDF5 with structure:
      layers/<layer_name>/vars/<index>
    For each model layer found in HDF5, call layer.set_weights([...]).
    Returns: (loaded_count, missing_list)
    """
    if not os.path.exists(h5_path):
        raise FileNotFoundError(h5_path)
    loaded = 0
    missing = []
    with h5py.File(h5_path, 'r') as f:
        if 'layers' not in f:
            raise ValueError("HDF5 file does not contain top-level 'layers' group.")
        layers_grp = f['layers']
        for layer in model.layers:
            lname = layer.name
            if lname in layers_grp:
                grp = layers_grp[lname]
                if 'vars' in grp:
                    var_grp = grp['vars']
                    # sort keys numerically if they are digits
                    keys = list(var_grp.keys())
                    try:
                        keys = sorted(keys, key=lambda x: int(x) if x.isdigit() else x)
                    except Exception:
                        keys = sorted(keys)
                    arrays = [np.array(var_grp[k]) for k in keys]
                    try:
                        # set weights (works for conv, dense, batchnorm)
                        layer.set_weights(arrays)
                        loaded += 1
                        print(f"[OK] Loaded weights for layer: {lname} (arrays: {len(arrays)})")
                    except Exception as e:
                        print(f"[ERR] Failed set_weights for layer {lname}: {e}")
                        missing.append(lname)
                else:
                    # non-param layers (pooling, dropout) may have no vars
                    print(f"[SKIP] Layer {lname} has no 'vars' group (likely non-param).")
            else:
                missing.append(lname)
    return loaded, missing

def prepare_input(image_path, target_size=(224,224)):
    """Load image or random data and return (orig_rgb, input_batch_float32)."""
    if image_path and os.path.exists(image_path):
        orig = cv2.imread(image_path)
        orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
        orig_resized = cv2.resize(orig, target_size)
        x = np.expand_dims(orig_resized.astype('float32') / 255.0, axis=0)
        return orig_resized, x
    else:
        print("Test image not found; using random input for sanity check.")
        x = np.random.rand(1, target_size[0], target_size[1], 3).astype('float32')
        orig = (x[0] * 255).astype('uint8')
        return orig, x

def compute_gradcam_and_save(model, grad_layer_name, input_batch, orig_rgb, out_path):
    """Compute Grad-CAM for given model and save overlay to out_path."""
    # Build grad model: conv outputs + preds
    conv_layer = model.get_layer(grad_layer_name)
    grad_model = keras.models.Model(inputs=model.input, outputs=[conv_layer.output, model.output])

    # compute gradients
    x_tensor = tf.convert_to_tensor(input_batch)
    with tf.GradientTape() as tape:
        tape.watch(x_tensor)
        convs, preds = grad_model(x_tensor, training=False)
        # if scalar output shape (None,1)
        class_score = preds[:, 0]
    grads = tape.gradient(class_score, convs)
    pooled_grads = tf.reduce_mean(grads, axis=(1,2))  # (1,channels)
    convs_np = convs[0].numpy()
    pooled_np = pooled_grads[0].numpy()

    # weight conv maps
    convs_np *= pooled_np[None, None, :]
    cam = np.sum(convs_np, axis=-1)
    cam = np.maximum(cam, 0)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    # overlay
    heatmap = (cam * 255).astype('uint8')
    heatmap_resized = cv2.resize(heatmap, (orig_rgb.shape[1], orig_rgb.shape[0]))
    heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(orig_rgb.astype('uint8'), 0.6, heatmap_color, 0.4, 0)
    # convert RGB -> BGR for cv2.imwrite
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_path, overlay_bgr)
    return preds.numpy(), out_path

def main():
    print("Building model with names matching HDF5...")
    model = build_model_matching_h5(input_shape=(224,224,3))
    print("Model built. Total layers:", len(model.layers))

    print("Loading weights from:", WEIGHTS_H5)
    loaded_count, missing = load_weights_layers_vars(model, WEIGHTS_H5)
    print(f"Loaded layers: {loaded_count}. Missing/failures: {len(missing)}")
    if missing:
        print("Missing/fail layers:", missing)
    # Show summary
    model.summary()

    # Save reconstructed model (both .keras and .h5)
    try:
        print("Saving reconstructed model to:", SAVE_PATH_KERAS)
        model.save(SAVE_PATH_KERAS)   # Keras native format
        print("Saved:", SAVE_PATH_KERAS)
    except Exception as e:
        print("Could not save .keras, error:", e)
    try:
        print("Also saving legacy HDF5 to:", SAVE_PATH_H5)
        model.save(SAVE_PATH_H5)
        print("Saved:", SAVE_PATH_H5)
    except Exception as e:
        print("Could not save .h5, error:", e)

    # Prepare input
    orig, x = prepare_input(TEST_IMAGE, target_size=(224,224))

    # Gradient sanity check
    # choose last conv layer name as 'conv2d_5' (per your HDF5)
    grad_layer_name = 'conv2d_5'
    try:
        preds, overlay_path = None, None
        preds, overlay_path = compute_gradcam_and_save(model, grad_layer_name, x, orig, os.path.join(OUT_DIR, 'gradcam_tb_overlay.png'))
        print("Prediction (raw):", preds)
        print("Grad-CAM overlay saved to:", overlay_path)
    except Exception as e:
        print("Grad-CAM failed:", e)
        # attempt to list available conv layers
        convs = [layer.name for layer in model.layers if hasattr(layer, 'output_shape') and len(layer.output_shape) == 4]
        print("Conv-like layers in model:", convs)

if __name__ == '__main__':
    main()
