import numpy as np
import streamlit as st
from config import CFG
from preprocessing import (
    crop_face_pil,
    normalize,
    extract_faces_from_video,
    extract_frame_features_for_lstm,
)


# ─────────────────────────────────────────────────────────────────────────────
# Model loaders
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def get_model():
    """Load and cache the CNN image model with proper warm-up."""
    import os
    if not os.path.exists(CFG.model_path):
        raise FileNotFoundError(
            f"Model not found at '{CFG.model_path}'.\n"
            "Run create_model.py to create a dummy model, or train.py to train one."
        )
    # Load with compile=True so BatchNorm moving averages are restored correctly
    from tensorflow.keras.models import load_model as keras_load
    import tensorflow as tf

    model = keras_load(
        CFG.model_path,
        compile=False,
        custom_objects=None,
    )

    # Re-compile so metrics work
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    # Warm-up: run two dummy passes so BatchNorm and Dropout
    # settle into inference mode properly
    dummy = np.zeros((1, CFG.face_size, CFG.face_size, 3), dtype="float32")
    model(dummy, training=False)
    model(dummy, training=False)

    return model


@st.cache_resource(show_spinner=False)
def get_cnn_extractor():
    """
    Load or derive the CNN feature extractor (backbone → GAP output).
    Used by the video LSTM pipeline.
    """
    import os
    import tensorflow as tf
    from tensorflow.keras.models import Model

    if os.path.exists(CFG.cnn_extractor_path):
        from tensorflow.keras.models import load_model as keras_load
        extractor = keras_load(CFG.cnn_extractor_path, compile=False)
    else:
        # Derive from the loaded CNN model:
        # outer model has layers: [Input, EfficientNetB4, GAP, BN, Dropout, Dense, ...]
        # We want: Input → EfficientNetB4 → GAP
        cnn = get_model()
        try:
            gap_out = cnn.get_layer("gap").output
        except ValueError:
            # fallback: 4th from last layer before the two Dense+Dropout head
            gap_out = cnn.layers[-4].output
        extractor = Model(inputs=cnn.input, outputs=gap_out,
                          name="CNN_Feature_Extractor")

    # Warm up
    dummy = np.zeros((1, CFG.face_size, CFG.face_size, 3), dtype="float32")
    extractor(dummy, training=False)

    return extractor


@st.cache_resource(show_spinner=False)
def get_lstm_model():
    """
    Load the BiLSTM video model.
    Returns None if not yet trained (caller falls back to CNN mean).
    """
    import os
    if not os.path.exists(CFG.lstm_model_path):
        return None
    from tensorflow.keras.models import load_model as keras_load
    return keras_load(CFG.lstm_model_path, compile=False)


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _predict_batch(faces_rgb, model):
    """
    Run CNN model on a batch of RGB face crops.
    Uses model.__call__ with training=False to ensure BatchNorm
    uses its stored moving averages (not batch statistics).
    Returns list of float scores.
    """
    import tensorflow as tf
    batch = np.stack([normalize(f) for f in faces_rgb]).astype("float32")
    preds = model(batch, training=False)          # <-- training=False is critical
    return preds.numpy().squeeze(-1).tolist()


def _make_result(score, frame_scores, faces, model_tag="CNN"):
    is_fake = score >= CFG.fake_threshold
    return {
        "label":        "FAKE" if is_fake else "REAL",
        "score":        round(float(score), 4),
        "confidence":   round(float(max(score, 1 - score)), 4),
        "is_fake":      is_fake,
        "frame_scores": [round(float(s), 4) for s in frame_scores],
        "faces":        faces,
        "n_faces":      len(faces),
        "model_tag":    model_tag,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Image prediction
# ─────────────────────────────────────────────────────────────────────────────

def predict_image(pil_image, model):
    """Predict on a single PIL Image using the CNN model."""
    face = crop_face_pil(pil_image)
    if face is None:
        return {
            "label": "UNKNOWN", "score": 0.5, "confidence": 0.0,
            "is_fake": False, "frame_scores": [],
            "faces": np.empty((0, CFG.face_size, CFG.face_size, 3), dtype=np.uint8),
            "n_faces": 0, "model_tag": "CNN",
            "error": "No face detected in the image.",
        }
    scores = _predict_batch([face], model)
    return _make_result(scores[0], scores, np.array([face]), model_tag="CNN")


# ─────────────────────────────────────────────────────────────────────────────
# Video prediction  (CNN + BiLSTM, falls back to CNN mean if LSTM unavailable)
# ─────────────────────────────────────────────────────────────────────────────

def predict_video(video_path, model, num_frames=CFG.num_frames):
    """
    Predict on a video using the CNN + BiLSTM pipeline.

    1. Extract MTCNN face crops from uniformly sampled frames.
    2. Get per-frame CNN scores (for the bar chart).
    3. Run CNN extractor → BiLSTM for temporal verdict.
       Falls back to CNN mean-pooling if LSTM not trained yet.
    """
    faces, _ = extract_faces_from_video(video_path, num_frames)

    if len(faces) == 0:
        return {
            "label": "UNKNOWN", "score": 0.5, "confidence": 0.0,
            "is_fake": False, "frame_scores": [],
            "faces": np.empty((0, CFG.face_size, CFG.face_size, 3), dtype=np.uint8),
            "n_faces": 0, "model_tag": "CNN+LSTM",
            "error": "No faces detected in any video frame.",
        }

    # Per-frame CNN scores for chart (always computed)
    frame_scores = _predict_batch(list(faces), model)

    # CNN + LSTM temporal prediction
    try:
        lstm_model = get_lstm_model()

        if lstm_model is not None:
            cnn_extractor = get_cnn_extractor()
            seq_tensor    = extract_frame_features_for_lstm(
                video_path, cnn_extractor, num_frames
            )
            if seq_tensor is not None:
                video_score = float(lstm_model(seq_tensor, training=False).numpy()[0][0])
                model_tag   = "CNN+LSTM"
            else:
                video_score = float(np.mean(frame_scores))
                model_tag   = "CNN (no faces for LSTM)"
        else:
            video_score = float(np.mean(frame_scores))
            model_tag   = "CNN (LSTM not trained yet)"

    except Exception as e:
        print(f"[WARN] LSTM inference failed ({e}). Falling back to CNN mean.")
        video_score = float(np.mean(frame_scores))
        model_tag   = "CNN (fallback)"

    return _make_result(video_score, frame_scores, faces, model_tag=model_tag)
