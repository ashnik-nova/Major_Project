# load_h5_by_layer_and_gradcam.py
import os
import h5py
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import cv2

# ========== UPDATE THESE PATHS ==========
WEIGHTS_H5 = r'B:\CodingStuffs\XAI_Project\Backend\models\dog_cat\json_weights\model.weights.h5'
TEST_IMAGE = r'B:\CodingStuffs\XAI_Project\Backend\static\dog.1002.jpg'   # replace with valid image path
OUT_DIR = r'B:\CodingStuffs\XAI_Project\Backend\gradcam_output'
os.makedirs(OUT_DIR, exist_ok=True)
# =======================================

# Rebuild model architecture exactly as you posted
def build_model(input_shape=(128,128,3)):
    m = keras.Sequential(name='sequential_1')
    m.add(layers.Conv2D(16, (3,3), activation='relu', padding='same', input_shape=input_shape, name='conv2d'))
    m.add(layers.MaxPooling2D((2,2), name='max_pooling2d'))
    m.add(layers.Dropout(0.0, name='dropout'))

    m.add(layers.Conv2D(32, (3,3), activation='relu', padding='same', name='conv2d_1'))
    m.add(layers.MaxPooling2D((2,2), name='max_pooling2d_1'))
    m.add(layers.Dropout(0.0, name='dropout_1'))

    m.add(layers.Conv2D(64, (3,3), activation='relu', padding='same', name='conv2d_2'))
    m.add(layers.MaxPooling2D((2,2), name='max_pooling2d_2'))
    m.add(layers.Dropout(0.0, name='dropout_2'))

    m.add(layers.Conv2D(128, (3,3), activation='relu', padding='same', name='conv2d_3'))
    m.add(layers.MaxPooling2D((2,2), name='max_pooling2d_3'))
    m.add(layers.Dropout(0.0, name='dropout_3'))

    m.add(layers.Conv2D(256, (3,3), activation='relu', padding='same', name='conv2d_4'))
    m.add(layers.MaxPooling2D((2,2), name='max_pooling2d_4'))
    m.add(layers.Dropout(0.0, name='dropout_4'))

    m.add(layers.Flatten(name='flatten'))
    m.add(layers.Dense(64, activation='relu', name='dense'))
    m.add(layers.Dense(32, activation='relu', name='dense_1'))
    m.add(layers.Dense(1, activation='sigmoid', name='dense_2'))
    return m

# Build model
model = build_model()
print("Model built. Layers:")
for i, layer in enumerate(model.layers):
    print(i, layer.name, getattr(layer, 'output_shape', None))

# Function to load weights from HDF5 with structure layers/<name>/vars/<i>
def load_weights_from_custom_h5(model, h5_path):
    if not os.path.exists(h5_path):
        raise FileNotFoundError(h5_path)
    loaded_count = 0
    missing = []
    with h5py.File(h5_path, 'r') as f:
        if 'layers' not in f:
            raise ValueError("HDF5 format unexpected: no top-level 'layers' group found.")
        layers_grp = f['layers']
        for layer in model.layers:
            lname = layer.name
            if lname in layers_grp:
                grp = layers_grp[lname]
                # Expect subgroup 'vars' containing datasets '0','1',...
                if 'vars' in grp:
                    var_grp = grp['vars']
                    # collect arrays in index order: '0', '1', ...
                    keys = sorted([k for k in var_grp.keys()], key=lambda x: int(x) if x.isdigit() else x)
                    weights = []
                    for k in keys:
                        arr = np.array(var_grp[k])
                        weights.append(arr)
                    try:
                        layer.set_weights(weights)
                        loaded_count += 1
                        print(f"Loaded weights for layer: {lname} (num params arrays: {len(weights)})")
                    except Exception as e:
                        print(f"Failed to set_weights for layer {lname}: {e}")
                        missing.append(lname)
                else:
                    # layer exists but no vars group (e.g., pooling, dropout)
                    print(f"Layer {lname} has no vars in h5 (likely non-param layer).")
            else:
                missing.append(lname)
    return loaded_count, missing

print("\nLoading weights from:", WEIGHTS_H5)
loaded_count, missing = load_weights_from_custom_h5(model, WEIGHTS_H5)
print(f"\nFinished loading. Layers loaded: {loaded_count}. Missing / failed: {len(missing)}")
if missing:
    print("Missing or failed layers:", missing)

# Sanity: show model summary and parameter count
model.summary()

# Prepare input image (match training preprocessing)
if os.path.exists(TEST_IMAGE):
    orig = cv2.imread(TEST_IMAGE)[:,:,::-1]  # BGR->RGB
    orig = cv2.resize(orig, (128,128))
    x = np.expand_dims(orig.astype('float32') / 255.0, axis=0)
else:
    print("Test image not found, using random data for sanity check.")
    x = np.random.rand(1,128,128,3).astype('float32')
    orig = (x[0]*255).astype('uint8')

# Create grad model to get conv outputs and preds
last_conv = model.get_layer('conv2d_4')
grad_model = keras.models.Model(inputs=model.input, outputs=[last_conv.output, model.output])

# Gradients sanity check wrt input
x_tensor = tf.convert_to_tensor(x)
with tf.GradientTape() as tape:
    tape.watch(x_tensor)
    convs, preds = grad_model(x_tensor, training=False)
    score = preds[:, 0]  # assuming binary sigmoid output
grads_wrt_input = tape.gradient(score, x_tensor)
print("Prediction:", preds.numpy(), "grads shape:", grads_wrt_input.shape, "grad norm:", tf.norm(grads_wrt_input).numpy())

# Compute Grad-CAM
def compute_gradcam_from_convs(img_tensor):
    img_t = tf.convert_to_tensor(img_tensor)
    with tf.GradientTape() as tape:
        tape.watch(img_t)
        convs, preds = grad_model(img_t, training=False)
        score = preds[:, 0]
    grads = tape.gradient(score, convs)  # shape (1,h,w,c)
    pooled = tf.reduce_mean(grads, axis=(1,2))  # (1,c)
    convs_np = convs[0].numpy()
    pooled_np = pooled[0].numpy()
    convs_np *= pooled_np[None, None, :]
    cam = np.sum(convs_np, axis=-1)
    cam = np.maximum(cam, 0)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam

heatmap = compute_gradcam_from_convs(x)
heatmap_resized = cv2.resize((heatmap * 255).astype('uint8'), (orig.shape[1], orig.shape[0]))
heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
overlay = cv2.addWeighted(orig.astype('uint8'), 0.6, heatmap_color, 0.4, 0)
out_path = os.path.join(OUT_DIR, 'gradcam_overlay_loaded.png')
cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
print("Saved Grad-CAM overlay to:", out_path)

# Save rebuilt + weight-loaded model in TF2.15-compatible format
model.save("dogcat_reconstructed.keras")     # Keras v3 format
model.save("dogcat_reconstructed.h5")        # H5 format (TF2.15 friendly)
