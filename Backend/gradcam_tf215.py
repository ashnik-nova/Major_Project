# wrap_savedmodel_and_gradcam.py
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import warnings

# --------- USER: update these paths ----------
SAVED_MODEL_DIR = r'B:\CodingStuffs\XAI_Project\Backend\models\dog_cat\savedmodel'   # zip -> folder path
TEST_IMAGE_PATH  = r'B:\CodingStuffs\XAI_Project\Backend\static\dog.1002.jpg'             # sample image to run gradcam on
OUT_DIR = 'gradcam_out'
os.makedirs(OUT_DIR, exist_ok=True)

# --------- Load low-level SavedModel (tf.Module/_UserObject) ----------
loaded = tf.saved_model.load(SAVED_MODEL_DIR)
# List signature names available
signatures = {}
try:
    signatures = {k: v for k, v in loaded.signatures.items()}
except Exception:
    # some SavedModels expose signatures differently; try attribute access
    try:
        signatures = { 'serve': loaded.signatures['serve'] }
    except Exception:
        signatures = {}

print("Available signatures (keys):", list(signatures.keys()))

# Prefer 'serve' then 'serving_default', otherwise use the first signature
sig_name = None
for candidate in ('serve', 'serving_default'):
    if candidate in signatures:
        sig_name = candidate
        break
if sig_name is None:
    # fallback: use the first signature available in signatures dict
    if len(signatures) > 0:
        sig_name = list(signatures.keys())[0]
    else:
        # if no signatures dict, try calling the Module directly (may be callable)
        # we'll try to inspect callable attributes
        sig_name = None

print("Using signature:", sig_name)

# Get the concrete function we'll call
if sig_name is not None:
    serve_fn = signatures[sig_name]  # ConcreteFunction
else:
    # fallback: try to find a callable attribute on the loaded module
    # many SavedModels expose a 'serve' or '__call__' behavior
    # We will attempt to use loaded.__call__ if present, else raise
    if callable(loaded):
        serve_fn = loaded
        print("Using module callable directly.")
    else:
        raise RuntimeError("No usable signature found in SavedModel and module is not callable. "
                           "Please re-export model with Keras-friendly format or share the SavedModel details.")

# Inspect signature inputs/outputs
try:
    print("Signature input_spec:", serve_fn.structured_input_signature)
    print("Signature outputs:", serve_fn.structured_outputs)
except Exception:
    # If serve_fn is not a ConcreteFunction but a callable, skip
    pass

# Build Keras wrapper that calls the signature and returns the first output tensor
# We must map input names/types to a Keras Input. We assume the model expects shape (None,128,128,3) float32
input_shape = (128, 128, 3)   # your model's input shape — matches what you exported
inp = tf.keras.Input(shape=input_shape, dtype=tf.float32, name='input_layer_1')

# ConcreteFunction expects batched Tensor(s). We call it and extract outputs robustly.
def call_signature(x):
    # ConcreteFunction expects a dict or positional args depending on signature.
    # We'll try both patterns.
    try:
        # If serve_fn is a ConcreteFunction: calling it returns a dict mapping names -> tensors
        out = serve_fn(x)  # ConcreteFunction supports direct call with Tensor
    except Exception:
        try:
            # Try calling with a dict keyed by input name (if known). We'll try common names:
            possible_keys = ['input_1', 'input_layer_1', 'args_0', 'inputs']
            called = None
            for k in possible_keys:
                try:
                    out = serve_fn(**{k: x})
                    called = True
                    break
                except Exception:
                    continue
            if called is None:
                # last resort: positional
                out = serve_fn(x)
        except Exception as e:
            raise RuntimeError(f"Failed to call SavedModel signature: {e}")

    # out can be: a Tensor, a dict of Tensors, or a list/tuple of Tensors
    if isinstance(out, dict):
        # get first value
        first_val = list(out.values())[0]
        return first_val
    elif isinstance(out, (list, tuple)):
        return out[0]
    else:
        # assume single Tensor
        return out

# Use a Lambda layer to call the signature (wrap in tf.function to avoid tracing overhead)
# Note: using a Lambda that calls serve_fn on a Keras Input may produce tracing warnings but works.
out_tensor = tf.keras.layers.Lambda(lambda x: call_signature(x))(inp)
keras_model = tf.keras.Model(inputs=inp, outputs=out_tensor)

# Print summary now
print("Wrapped Keras model summary:")
try:
    keras_model.summary()
except Exception as e:
    warnings.warn(f"Could not print summary: {e}")

# --------- Now we can use GradientTape on keras_model ----------
# Prepare a test image (match training preprocessing)
orig = cv2.imread(TEST_IMAGE_PATH)[:,:,::-1]  # BGR->RGB
orig_resized = cv2.resize(orig, (input_shape[1], input_shape[0]))
x = orig_resized.astype('float32') / 255.0   # change if different preprocessing
x = np.expand_dims(x, axis=0)

# Sanity check: forward pass
pred = keras_model(x, training=False).numpy()
print("Model prediction (raw):", pred, "shape:", pred.shape)

# Build grad_model to return conv outputs + preds.
# Find the last conv layer in the wrapped model by searching layers (same names as original likely preserved)
last_conv_layer_name = None
for layer in reversed(keras_model.layers):
    try:
        shp = layer.output_shape
        if isinstance(shp, tuple) and len(shp) == 4:
            last_conv_layer_name = layer.name
            break
    except Exception:
        continue

print("Detected last conv layer in wrapper:", last_conv_layer_name)
if last_conv_layer_name is None:
    raise RuntimeError("Could not find a convolutional layer in the wrapped model. "
                       "You may need to manually set last_conv_layer_name to e.g. 'conv2d_4'")

last_conv = keras_model.get_layer(last_conv_layer_name)
grad_model = tf.keras.models.Model(inputs=keras_model.input, outputs=[last_conv.output, keras_model.output])

# Grad-CAM compute (binary output assumed, picks output index 0)
def compute_gradcam(img_tensor):
    img_t = tf.convert_to_tensor(img_tensor)
    with tf.GradientTape() as tape:
        tape.watch(img_t)
        convs, preds = grad_model(img_t, training=False)
        # for single-output (None,1), pick preds[:,0]
        class_score = preds[:, 0]

    grads = tape.gradient(class_score, convs)   # shape (1,Hc,Wc,C)
    pooled_grads = tf.reduce_mean(grads, axis=(1,2))  # (1,C)
    convs = convs[0]     # (Hc,Wc,C)
    pooled = pooled_grads[0]

    convs_weighted = convs * pooled[None, None, :]
    cam = tf.reduce_sum(convs_weighted, axis=-1)
    cam = tf.nn.relu(cam)
    cam = cam - tf.reduce_min(cam)
    cam = cam / (tf.reduce_max(cam) + 1e-8)
    return cam.numpy()

# Compute and save overlay
heatmap = compute_gradcam(x)
# Resize heatmap to original image
h, w = orig.shape[:2]
heatmap_resized = cv2.resize((heatmap * 255).astype('uint8'), (w, h))
heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
overlay = cv2.addWeighted(orig.astype('uint8'), 0.6, heatmap_color, 0.4, 0)

outpath = os.path.join(OUT_DIR, 'gradcam_overlay.png')
cv2.imwrite(outpath, overlay[:,:,::-1])  # RGB->BGR for imwrite
print("Saved Grad-CAM overlay to:", outpath)
