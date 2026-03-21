import numpy as np
import streamlit as st
from config import CFG
from preprocessing import crop_face_pil, normalize, extract_faces_from_video

@st.cache_resource(show_spinner=False)
def get_model():
    """Load and cache model for the Streamlit session."""
    import os
    if not os.path.exists(CFG.model_path):
        raise FileNotFoundError(
            f"Model not found at '{CFG.model_path}'.\n"
            "Train it with train.py or download a pretrained .h5 file."
        )
    from model import load_model
    return load_model()

def _predict_batch(faces_rgb, model):
    """Run model on a batch of RGB face crops. Returns list of scores."""
    batch = np.stack([normalize(f) for f in faces_rgb])
    preds = model.predict(batch, verbose=0)
    return preds.squeeze(-1).tolist()

def _make_result(score, frame_scores, faces):
    is_fake = score >= CFG.fake_threshold
    return {
        "label":        "FAKE" if is_fake else "REAL",
        "score":        round(float(score), 4),
        "confidence":   round(float(max(score, 1 - score)), 4),
        "is_fake":      is_fake,
        "frame_scores": [round(float(s), 4) for s in frame_scores],
        "faces":        faces,
        "n_faces":      len(faces),
    }

def predict_image(pil_image, model):
    """Predict on a single PIL Image."""
    face = crop_face_pil(pil_image)
    if face is None:
        return {
            "label": "UNKNOWN", "score": 0.5, "confidence": 0.0,
            "is_fake": False, "frame_scores": [],
            "faces": np.empty((0, CFG.face_size, CFG.face_size, 3), dtype=np.uint8),
            "n_faces": 0, "error": "No face detected in the image.",
        }
    scores = _predict_batch([face], model)
    return _make_result(scores[0], scores, np.array([face]))

def predict_video(video_path, model, num_frames=CFG.num_frames):
    """Predict on a video file."""
    faces, _ = extract_faces_from_video(video_path, num_frames)
    if len(faces) == 0:
        return {
            "label": "UNKNOWN", "score": 0.5, "confidence": 0.0,
            "is_fake": False, "frame_scores": [],
            "faces": np.empty((0, CFG.face_size, CFG.face_size, 3), dtype=np.uint8),
            "n_faces": 0, "error": "No faces detected in any video frame.",
        }
    frame_scores = _predict_batch(faces, model)
    video_score  = float(np.mean(frame_scores))
    return _make_result(video_score, frame_scores, faces)