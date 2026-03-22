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
    """Load, warm up, and cache the CNN image model."""
    import os
    if not os.path.exists(CFG.model_path):
        raise FileNotFoundError(
            f"Model not found at '{CFG.model_path}'.\n"
            "Run create_model.py or train.py to create it first."
        )
    from model import load_model
    model = load_model()

    # Warm-up pass — forces Keras to fully build all internal states.
    # Without this, BatchNormalization layers can output garbage on the
    # first real inference call, causing everything to look FAKE.
    dummy = np.zeros((1, CFG.face_size, CFG.face_size, 3), dtype="float32")
    model.predict(dummy, verbose=0)

    return model


@st.cache_resource(show_spinner=False)
def get_cnn_extractor():
    """
    Load the CNN feature extractor for the video LSTM pipeline.
    Tries CFG.cnn_extractor_path first, then falls back to deriving
    it from the main CNN model's GAP layer.
    """
    import os
    from tensorflow.keras.models import load_model as keras_load
    from tensorflow.keras.models import Model

    if os.path.exists(CFG.cnn_extractor_path):
        extractor = keras_load(CFG.cnn_extractor_path, compile=False)
    else:
        # Derive from the main model
        cnn = get_model()
        try:
            gap_out = cnn.get_layer("gap").output
        except ValueError:
            # Fall back: use output of the layer before the Dense head
            gap_out = cnn.layers[-4].output
        extractor = Model(inputs=cnn.input, outputs=gap_out,
                          name="CNN_Feature_Extractor")

    # Warm up
    dummy = np.zeros((1, CFG.face_size, CFG.face_size, 3), dtype="float32")
    extractor.predict(dummy, verbose=0)

    return extractor


@st.cache_resource(show_spinner=False)
def get_lstm_model():
    """
    Load the BiLSTM video model from CFG.lstm_model_path.
    Returns None (gracefully) if the file does not exist yet.
    """
    import os
    from tensorflow.keras.models import load_model as keras_load

    if not os.path.exists(CFG.lstm_model_path):
        return None   # caller checks for None and falls back to CNN mean

    model = keras_load(CFG.lstm_model_path, compile=False)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _predict_batch(faces_rgb, model):
    """Run CNN model on a batch of RGB face crops. Returns list of float scores."""
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
        "model_tag":    model_tag,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Image prediction  (CNN only)
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

    Steps:
      1. Extract MTCNN face crops from uniformly sampled frames.
      2. Get per-frame CNN scores (shown in bar chart).
      3. Run CNN extractor + BiLSTM for the final temporal verdict.
         Falls back to CNN mean-pooling if LSTM model is unavailable.
    """
    # Step 1: extract face crops
    faces, _ = extract_faces_from_video(video_path, num_frames)

    if len(faces) == 0:
        return {
            "label": "UNKNOWN", "score": 0.5, "confidence": 0.0,
            "is_fake": False, "frame_scores": [],
            "faces": np.empty((0, CFG.face_size, CFG.face_size, 3), dtype=np.uint8),
            "n_faces": 0, "model_tag": "CNN+LSTM",
            "error": "No faces detected in any video frame.",
        }

    # Step 2: per-frame CNN scores for the chart
    frame_scores = _predict_batch(list(faces), model)

    # Step 3: CNN + LSTM temporal prediction
    try:
        lstm_model = get_lstm_model()

        if lstm_model is not None:
            cnn_extractor = get_cnn_extractor()
            seq_tensor    = extract_frame_features_for_lstm(
                video_path, cnn_extractor, num_frames
            )
            if seq_tensor is not None:
                video_score = float(lstm_model.predict(seq_tensor, verbose=0)[0][0])
                model_tag   = "CNN+LSTM"
            else:
                video_score = float(np.mean(frame_scores))
                model_tag   = "CNN (no faces for LSTM)"
        else:
            # lstm_model.h5 not trained yet — use CNN mean
            video_score = float(np.mean(frame_scores))
            model_tag   = "CNN (LSTM not trained)"

    except Exception as e:
        print(f"[WARN] LSTM inference failed ({e}). Falling back to CNN mean.")
        video_score = float(np.mean(frame_scores))
        model_tag   = "CNN (fallback)"

    return _make_result(video_score, frame_scores, faces, model_tag=model_tag)
