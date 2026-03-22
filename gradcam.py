import numpy as np
import tensorflow as tf
from config import CFG
from preprocessing import normalize


def _find_last_conv_layer(model):
    """
    Recursively search for the last Conv2D layer, including inside sub-models
    (e.g. EfficientNetB4 is a nested Model inside the outer classifier).
    Returns the layer object or None.
    """
    last_conv = None

    def _search(m):
        nonlocal last_conv
        for layer in m.layers:
            # Recurse into nested models
            if isinstance(layer, tf.keras.Model):
                _search(layer)
            elif isinstance(layer, tf.keras.layers.Conv2D):
                last_conv = layer

    _search(model)
    return last_conv


def compute_gradcam(model, face_rgb):
    import cv2

    # ── Step 1: find last Conv2D (searches nested sub-models) ────────────────
    last_conv_layer = _find_last_conv_layer(model)
    if last_conv_layer is None:
        raise ValueError("No Conv2D layer found for Grad-CAM")

    # ── Step 2: build grad model using the found layer ────────────────────────
    # We need to find which model actually owns this layer so we can
    # correctly wire the grad model inputs/outputs.
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[last_conv_layer.output, model.output]
    )

    # ── Step 3: preprocess ───────────────────────────────────────────────────
    img = normalize(face_rgb.astype("float32"))[np.newaxis]

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)

    # ── Step 4: compute weighted heatmap ─────────────────────────────────────
    pooled_grads  = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs  = conv_outputs[0]
    heatmap       = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap       = tf.squeeze(heatmap)
    heatmap       = np.maximum(heatmap.numpy(), 0)

    if np.max(heatmap) > 0:
        heatmap /= np.max(heatmap)

    # ── Step 5: resize + colormap ────────────────────────────────────────────
    H, W    = face_rgb.shape[:2]
    heatmap = cv2.resize(heatmap, (W, H))
    heatmap = (heatmap * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)


def overlay_heatmap(face_rgb, heatmap_rgb):
    blended = (
        (1 - CFG.gradcam_alpha) * face_rgb.astype("float32") +
        CFG.gradcam_alpha       * heatmap_rgb.astype("float32")
    )
    return np.clip(blended, 0, 255).astype(np.uint8)
