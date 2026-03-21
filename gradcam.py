import numpy as np
import tensorflow as tf
from config import CFG
from preprocessing import normalize

def compute_gradcam(model, face_rgb):
    import cv2

    # 🔥 STEP 1 — find EfficientNet base model
    base_model = None
    for layer in model.layers:
        if "efficientnet" in layer.name.lower():
            base_model = layer
            break

    if base_model is None:
        raise ValueError("EfficientNet base model not found")

    # 🔥 STEP 2 — get last conv layer
    last_conv_layer = base_model.get_layer("top_conv")

    # 🔥 STEP 3 — create grad model
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[last_conv_layer.output, model.output]
    )

    # 🔥 STEP 4 — preprocess input
    img = normalize(face_rgb.astype("float32"))[np.newaxis]
    img_tensor = tf.cast(img, tf.float32)

    # 🔥 STEP 5 — compute gradients
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)

    # 🔥 STEP 6 — global average pooling
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) > 0:
        heatmap /= np.max(heatmap)

    # 🔥 STEP 7 — resize + color
    H, W = face_rgb.shape[:2]
    heatmap = cv2.resize(heatmap.numpy(), (W, H))
    heatmap = (heatmap * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    return cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

def overlay_heatmap(face_rgb, heatmap_rgb):
    blended = (1 - CFG.gradcam_alpha) * face_rgb.astype("float32") + \
               CFG.gradcam_alpha * heatmap_rgb.astype("float32")
    return np.clip(blended, 0, 255).astype(np.uint8)
