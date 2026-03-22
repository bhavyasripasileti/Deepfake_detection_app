import os
import numpy as np
from config import CFG

_detector = None
_cv2 = None


def _get_cv2():
    global _cv2
    if _cv2 is None:
        import cv2
        _cv2 = cv2
    return _cv2


def get_detector():
    global _detector
    if _detector is None:
        from mtcnn import MTCNN
        _detector = MTCNN()
    return _detector


def crop_face(frame_bgr):
    cv2 = _get_cv2()
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    detections = get_detector().detect_faces(rgb)
    if not detections:
        return None
    best = max(detections, key=lambda d: d["confidence"])
    if best["confidence"] < 0.90:
        return None
    x, y, w, h = best["box"]
    margin_x = int(w * 0.2)
    margin_y = int(h * 0.2)
    H, W = rgb.shape[:2]
    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)
    x2 = min(W, x + w + margin_x)
    y2 = min(H, y + h + margin_y)
    crop = rgb[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (CFG.face_size, CFG.face_size))


def crop_face_pil(pil_image):
    cv2 = _get_cv2()
    rgb = np.array(pil_image.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return crop_face(bgr)


def normalize(face_rgb):
    from tensorflow.keras.applications.efficientnet import preprocess_input
    return preprocess_input(face_rgb.astype("float32"))


def extract_faces_from_video(video_path, num_frames=CFG.num_frames):
    """
    Extract up to num_frames face crops from a video.
    Returns (faces_array, frame_indices).
    Used by the CNN-only inference path (predict_video legacy) and
    also as the first step of the CNN+LSTM pipeline.
    """
    cv2 = _get_cv2()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return np.empty((0, CFG.face_size, CFG.face_size, 3), dtype=np.uint8), []

    total = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
    candidates = np.linspace(0, total - 1, num_frames * 3, dtype=int).tolist()
    faces, indices = [], []

    for idx in candidates:
        if len(faces) >= num_frames:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        face = crop_face(frame)
        if face is not None:
            faces.append(face)
            indices.append(int(idx))

    cap.release()

    if not faces:
        return np.empty((0, CFG.face_size, CFG.face_size, 3), dtype=np.uint8), []

    return np.stack(faces), indices


def extract_frame_features_for_lstm(video_path, cnn_extractor, num_frames=CFG.num_frames):
    """
    CNN + LSTM video pipeline — Step 1:
    Extract face crops from video → normalize → run through frozen CNN feature
    extractor → return feature sequence ready for the LSTM.

    Args:
        video_path:     path to video file.
        cnn_extractor:  Keras Model with output shape (batch, feature_dim).
                        Build with model.build_cnn_feature_extractor().
        num_frames:     sequence length (pads/truncates to exactly this many).

    Returns:
        np.ndarray of shape (1, num_frames, feature_dim)  — batch-ready tensor.
        Returns None if no faces could be detected.
    """
    faces, _ = extract_faces_from_video(video_path, num_frames)

    if len(faces) == 0:
        return None

    feature_seq = []
    for face_rgb in faces:
        normed = normalize(face_rgb)                          # (H, W, 3) float32
        tensor = np.expand_dims(normed, axis=0)               # (1, H, W, 3)
        feat   = cnn_extractor.predict(tensor, verbose=0)     # (1, feature_dim)
        feature_seq.append(feat[0])

    # Pad by repeating last frame if we got fewer than num_frames faces
    while len(feature_seq) < num_frames:
        feature_seq.append(feature_seq[-1])

    feature_seq = feature_seq[:num_frames]                    # trim if over

    # (1, num_frames, feature_dim)
    return np.expand_dims(np.stack(feature_seq, axis=0), axis=0)
