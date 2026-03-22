import numpy as np
import tensorflow as tf
from config import CFG
from preprocessing import normalize


def compute_gradcam(model, face_rgb):
    import cv2

    # ── Step 1: find the EfficientNetB4 sub-model ────────────────────────────
    backbone = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            backbone = layer
            break

    if backbone is None:
        raise ValueError("Could not find backbone sub-model inside classifier.")

    # ── Step 2: find last Conv2D inside backbone ──────────────────────────────
    last_conv = None
    for layer in backbone.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv = layer

    if last_conv is None:
        raise ValueError("No Conv2D layer found inside backbone.")

    # ── Step 3: build backbone_partial: backbone.input → last_conv.output ─────
    backbone_partial = tf.keras.Model(
        inputs=backbone.input,
        outputs=last_conv.output,
        name="backbone_partial"
    )

    # ── Step 4: GradientTape over the IMAGE tensor ────────────────────────────
    # We watch the input image and compute gradients of the loss
    # w.r.t. the conv feature map activation.
    img        = normalize(face_rgb.astype("float32"))[np.newaxis]
    img_tensor = tf.Variable(tf.cast(img, tf.float32))

    with tf.GradientTape() as tape:
        # backbone_partial uses backbone.input (its own input),
        # so pass img_tensor through it directly
        conv_outputs = backbone_partial(img_tensor, training=False)
        predictions  = model(img_tensor, training=False)
        loss         = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)

    if grads is None:
        raise ValueError("Gradients are None — model may not be connected properly.")

    # ── Step 5: weighted heatmap ──────────────────────────────────────────────
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out_val = conv_outputs[0]
    heatmap      = conv_out_val @ pooled_grads[..., tf.newaxis]
    heatmap      = tf.squeeze(heatmap).numpy()
    heatmap      = np.maximum(heatmap, 0)

    if np.max(heatmap) > 0:
        heatmap /= np.max(heatmap)

    # ── Step 6: resize + colormap ─────────────────────────────────────────────
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
