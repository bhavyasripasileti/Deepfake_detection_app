import cv2
import numpy as np
import tensorflow as tf
from config import CFG
from preprocessing import normalize

def compute_gradcam(model, face_rgb):
    """Compute Grad-CAM heatmap for a face crop. Returns RGB heatmap."""
    try:
        conv_layer = model.get_layer(CFG.gradcam_layer)
    except ValueError:
        # Search inside sub-models (e.g. EfficientNet wrapped in functional model)
        conv_layer = None
        for layer in model.layers:
            if hasattr(layer, "layers"):
                for sub in layer.layers:
                    if sub.name == CFG.gradcam_layer:
                        conv_layer = sub
                        break
    if conv_layer is None:
        raise ValueError(f"Layer '{CFG.gradcam_layer}' not found in model.")

    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[conv_layer.output, model.output],
    )
    img = normalize(face_rgb.astype("float32"))[np.newaxis]
    img_tensor = tf.cast(img, tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        conv_out, pred = grad_model(img_tensor, training=False)
        loss = pred[:, 0]

    grads       = tape.gradient(loss, conv_out)
    pooled      = tf.reduce_mean(grads, axis=(0, 1, 2))
    weighted    = conv_out[0] * pooled
    cam         = tf.reduce_mean(weighted, axis=-1).numpy()
    cam         = np.maximum(cam, 0)
    if cam.max() > 0:
        cam = cam / cam.max()

    H, W        = face_rgb.shape[:2]
    cam_resized = cv2.resize(cam, (W, H))
    cam_uint8   = (cam_resized * 255).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    return cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

def overlay_heatmap(face_rgb, heatmap_rgb):
    """Blend face crop with Grad-CAM heatmap."""
    blended = (1 - CFG.gradcam_alpha) * face_rgb.astype("float32") + \
               CFG.gradcam_alpha * heatmap_rgb.astype("float32")
    return np.clip(blended, 0, 255).astype(np.uint8)