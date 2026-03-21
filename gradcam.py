import numpy as np
import tensorflow as tf
from config import CFG
from preprocessing import normalize

def compute_gradcam(model, face_rgb):
    import cv2

    # 🔥 STEP 1 — find last conv layer dynamically
    last_conv_layer = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer
            break

    if last_conv_layer is None:
        raise ValueError("No Conv2D layer found for Grad-CAM")

    # 🔥 STEP 2 — build grad model (FULL graph, no break)
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[last_conv_layer.output, model.output]
    )

    # 🔥 STEP 3 — preprocess
    img = normalize(face_rgb.astype("float32"))[np.newaxis]

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)

    # 🔥 STEP 4 — compute weights
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) > 0:
        heatmap /= np.max(heatmap)

    # 🔥 STEP 5 — resize + color
    H, W = face_rgb.shape[:2]
    heatmap = cv2.resize(heatmap.numpy(), (W, H))
    heatmap = (heatmap * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    return cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

def overlay_heatmap(face_rgb, heatmap_rgb):
    blended = (1 - CFG.gradcam_alpha) * face_rgb.astype("float32") + \
               CFG.gradcam_alpha * heatmap_rgb.astype("float32")
    return np.clip(blended, 0, 255).astype(np.uint8)
