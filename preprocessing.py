import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "0"
import cv2
import numpy as np
from mtcnn import MTCNN
from config import CFG

_detector = None

def get_detector():
    global _detector
    if _detector is None:
        _detector = MTCNN()
    return _detector

def crop_face(frame_bgr):
    """Detect and crop face from BGR frame. Returns RGB crop or None."""
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
    """Detect and crop face from PIL Image."""
    rgb = np.array(pil_image.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return crop_face(bgr)

def normalize(face_rgb):
    """Apply EfficientNet preprocessing. Input: RGB uint8. Output: float32."""
    from tensorflow.keras.applications.efficientnet import preprocess_input
    return preprocess_input(face_rgb.astype("float32"))

def extract_faces_from_video(video_path, num_frames=CFG.num_frames):
    """Sample faces uniformly from a video. Returns (faces_array, indices)."""
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
