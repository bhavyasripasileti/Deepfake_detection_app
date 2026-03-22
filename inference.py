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
# Model loaders (cached per Streamlit session)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def get_model():
    """Load and cache the CNN image model."""
    import os
    if not os.path.exists(CFG.model_path):
        raise FileNotFoundError(
            f"Model not found at '{CFG.model_path}'.\n"
            "Run create_model.py or train.py to create it first."
        )
    from model import load_model
    return load_model()


@st.cache_resource(show_spinner=False)
def get_cnn_extractor():
    """
    Load and cache the CNN feature extractor used by the video LSTM pipeline.
    Expects weights/cnn_extractor.h5 (created by create_model.py).
    Falls back gracefully to building from the CNN image model if not found.
    """
    import os
    from tensorflow.keras.models import load_model as keras_load

    extractor_path = "weights/cnn_extractor.h5"
    if os.path.exists(extractor_path):
        return keras_load(extractor_path, compile=False)

    # Fallback: derive extractor from the main CNN model
    cnn = get_model()
    from tensorflow.keras.models import Model
    # Re-expose the GAP layer output as the extractor output
    try:
        feat_out  = cnn.get_layer("gap").output
    except ValueError:
        # If layer wasn't named 'gap', use the second-to-last layer
        feat_out  = cnn.layers[-3].output
    extractor = Model(inputs=cnn.input, outputs=feat_out, name="CNN_Feature_Extractor")
    return extractor


@st.cache_resource(show_spinner=False)
def get_lstm_model():
    """
    Load and cache the BiLSTM video model.
    Expects weights/lstm_model.h5 (created by create_model.py / train.py).
    """
    import os
    from tensorflow.keras.models import load_model as keras_load

    lstm_path = "weights/lstm_model.h5"
    if not os.path.exists(lstm_path):
        raise FileNotFoundError(
            f"LSTM model not found at '{lstm_path}'.\n"
            "Run create_model.py or train.py --video to create it first."
        )
    return keras_load(lstm_path, compile=False)


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _predict_batch(faces_rgb, model):
    """Run CNN model on a batch of RGB face crops. Returns list of scores."""
    batch = np.stack([normalize(f) for f in faces_rgb])
    preds = model.predict(batch, verbose=0)
    return preds.squeeze(-1).tolist()


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
        "model_tag":    model_tag,   # "CNN" or "CNN+LSTM" shown in UI
    }


# ─────────────────────────────────────────────────────────────────────────────
# Image prediction  (CNN only — unchanged)
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
# Video prediction  (CNN + BiLSTM — upgraded)
# ─────────────────────────────────────────────────────────────────────────────

def predict_video(video_path, model, num_frames=CFG.num_frames):
    """
    Predict on a video using the CNN + BiLSTM pipeline.

    Steps:
      1. Extract MTCNN face crops from uniformly sampled frames.
      2. Normalize each face and run it through the frozen CNN feature extractor
         to get a (feature_dim,) vector per frame.
      3. Stack into a sequence tensor (1, num_frames, feature_dim).
      4. Pass through the BiLSTM model for a single fake probability.
      5. Also store per-frame CNN scores (for the frame score chart in the UI).

    Falls back to CNN-only mean-pooling if the LSTM model is unavailable.

    Args:
        video_path: path to the uploaded video file.
        model:      CNN image model (used for per-frame scores & fallback).
        num_frames: number of frames to sample.

    Returns:
        Result dict compatible with the existing app.py UI expectations.
    """
    # ── Step 1: extract face crops ────────────────────────────────────────────
    faces, _ = extract_faces_from_video(video_path, num_frames)

    if len(faces) == 0:
        return {
            "label": "UNKNOWN", "score": 0.5, "confidence": 0.0,
            "is_fake": False, "frame_scores": [],
            "faces": np.empty((0, CFG.face_size, CFG.face_size, 3), dtype=np.uint8),
            "n_faces": 0, "model_tag": "CNN+LSTM",
            "error": "No faces detected in any video frame.",
        }

    # ── Step 2: per-frame CNN scores (kept for UI frame score chart) ──────────
    frame_scores = _predict_batch(list(faces), model)

    # ── Step 3 & 4: CNN + LSTM temporal prediction ────────────────────────────
    try:
        cnn_extractor = get_cnn_extractor()
        lstm_model    = get_lstm_model()

        seq_tensor = extract_frame_features_for_lstm(
            video_path, cnn_extractor, num_frames
        )

        if seq_tensor is not None:
            lstm_prob   = float(lstm_model.predict(seq_tensor, verbose=0)[0][0])
            video_score = lstm_prob
            model_tag   = "CNN+LSTM"
        else:
            # No faces found by extractor — fall back to CNN mean
            video_score = float(np.mean(frame_scores))
            model_tag   = "CNN (fallback)"

    except (FileNotFoundError, Exception) as e:
        # LSTM model unavailable — graceful fallback to CNN mean pooling
        print(f"[WARN] LSTM path unavailable ({e}). Falling back to CNN mean.")
        video_score = float(np.mean(frame_scores))
        model_tag   = "CNN (fallback)"

    return _make_result(video_score, frame_scores, faces, model_tag=model_tag)
