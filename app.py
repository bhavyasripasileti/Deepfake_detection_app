"""
Deepfake Face Detection Web Application
CNN (EfficientNet) + LSTM Architecture
- Image: CNN only
- Video: CNN (frame feature extractor) + LSTM (temporal classifier)
"""

import streamlit as st
import numpy as np
import cv2
import tempfile
import os
from PIL import Image
import time

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DeepFake Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); }

    .main-header {
        text-align: center;
        padding: 2rem 0 1rem;
    }
    .main-header h1 {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .main-header p {
        color: #94a3b8;
        font-size: 1.1rem;
    }

    .upload-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 2rem;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
    }

    .result-real {
        background: linear-gradient(135deg, rgba(52,211,153,0.15), rgba(16,185,129,0.1));
        border: 2px solid #34d399;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
    }
    .result-fake {
        background: linear-gradient(135deg, rgba(239,68,68,0.15), rgba(220,38,38,0.1));
        border: 2px solid #ef4444;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
    }
    .result-label {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .confidence-text {
        font-size: 1.2rem;
        color: #cbd5e1;
    }

    .model-badge {
        display: inline-block;
        background: rgba(167,139,250,0.2);
        border: 1px solid #a78bfa;
        border-radius: 20px;
        padding: 0.3rem 1rem;
        font-size: 0.85rem;
        color: #a78bfa;
        margin: 0.5rem 0;
    }

    .info-box {
        background: rgba(96,165,250,0.1);
        border-left: 4px solid #60a5fa;
        border-radius: 0 8px 8px 0;
        padding: 1rem 1.5rem;
        margin: 1rem 0;
        color: #cbd5e1;
        font-size: 0.95rem;
    }

    .frame-grid {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        justify-content: center;
        margin: 1rem 0;
    }

    .stProgress > div > div { background: linear-gradient(90deg, #a78bfa, #60a5fa); }

    div[data-testid="stFileUploader"] {
        background: rgba(255,255,255,0.03);
        border-radius: 12px;
        padding: 0.5rem;
    }

    .stButton > button {
        background: linear-gradient(135deg, #a78bfa, #60a5fa);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        width: 100%;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.85; }

    .architecture-info {
        background: rgba(255,255,255,0.04);
        border-radius: 12px;
        padding: 1.2rem;
        margin: 0.5rem 0;
        color: #94a3b8;
        font-size: 0.9rem;
        line-height: 1.7;
    }
</style>
""", unsafe_allow_html=True)

# ─── Lazy imports (heavy libs only when needed) ───────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    """Load CNN image model and CNN+LSTM video model."""
    from model import build_cnn_image_model, build_cnn_lstm_video_model
    cnn_model   = build_cnn_image_model()
    lstm_model  = build_cnn_lstm_video_model()
    return cnn_model, lstm_model

# ─── Prediction Helpers ───────────────────────────────────────────────────────
def predict_image(model, img_array: np.ndarray) -> tuple[float, str]:
    """Run CNN prediction on a single pre-processed image."""
    from predict import preprocess_image, classify
    tensor = preprocess_image(img_array)
    prob   = float(model.predict(tensor, verbose=0)[0][0])
    return prob, classify(prob)


def predict_video(cnn_model, lstm_model, video_path: str,
                  n_frames: int = 20, face_only: bool = True) -> tuple[float, str, list]:
    """
    Extract n_frames from video → CNN features → LSTM → real/fake.
    Returns (probability, label, list_of_frame_bgr_arrays).
    """
    from predict import preprocess_image, classify, extract_frames
    frames_bgr = extract_frames(video_path, n_frames)
    if len(frames_bgr) == 0:
        return 0.5, "Unknown", []

    # CNN feature extraction per frame  [n_frames, feature_dim]
    feature_seq = []
    for frame in frames_bgr:
        tensor  = preprocess_image(frame)
        feat    = cnn_model.predict(tensor, verbose=0)   # (1, feature_dim)
        feature_seq.append(feat[0])

    # Pad / trim to exactly n_frames
    while len(feature_seq) < n_frames:
        feature_seq.append(feature_seq[-1])
    feature_seq = feature_seq[:n_frames]

    # Stack → (1, n_frames, feature_dim)
    seq_tensor = np.expand_dims(np.stack(feature_seq, axis=0), axis=0)
    prob       = float(lstm_model.predict(seq_tensor, verbose=0)[0][0])
    return prob, classify(prob), frames_bgr


# ─── UI ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🔍 DeepFake Detector</h1>
    <p>Powered by <strong>EfficientNet B0</strong> + <strong>BiLSTM</strong> — detect manipulated faces in images & videos</p>
</div>
""", unsafe_allow_html=True)

# Architecture info
col_a, col_b = st.columns(2)
with col_a:
    st.markdown("""
    <div class="architecture-info">
        <strong style="color:#a78bfa">🖼️ Image Pipeline</strong><br>
        EfficientNet B0 → Global Average Pooling → Dense → <em>Real / Fake</em>
    </div>
    """, unsafe_allow_html=True)
with col_b:
    st.markdown("""
    <div class="architecture-info">
        <strong style="color:#60a5fa">🎬 Video Pipeline</strong><br>
        EfficientNet B0 (feature extractor, frozen) → sequence of frame features → BiLSTM → <em>Real / Fake</em>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ─── Sidebar settings ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    n_frames = st.slider("Frames to sample (video)", 8, 40, 20, 2,
                         help="More frames = slower but more accurate")
    threshold = st.slider("Fake detection threshold", 0.3, 0.7, 0.5, 0.05,
                          help="Lower = stricter fake detection")
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
    **Model Architecture**
    - Feature extractor: EfficientNet B0
    - Temporal model: Bidirectional LSTM
    - Input size: 224 × 224
    - Sequence length: configurable

    **Supported formats**
    - Images: JPG, JPEG, PNG, WEBP
    - Videos: MP4, AVI, MOV, MKV
    """)

# ─── Upload ───────────────────────────────────────────────────────────────────
st.markdown('<div class="upload-card">', unsafe_allow_html=True)
tab_img, tab_vid = st.tabs(["🖼️ Image Detection", "🎬 Video Detection"])

# ════════════════════ IMAGE TAB ═══════════════════════════════════════════════
with tab_img:
    st.markdown("#### Upload an image to check for deepfakes")
    uploaded_img = st.file_uploader(
        "Choose image", type=["jpg","jpeg","png","webp"],
        key="img_upload", label_visibility="collapsed"
    )

    if uploaded_img:
        col1, col2 = st.columns([1, 1])
        with col1:
            image = Image.open(uploaded_img).convert("RGB")
            st.image(image, caption="Uploaded Image", use_column_width=True)

        with col2:
            if st.button("🔍 Analyze Image", key="btn_img"):
                with st.spinner("Loading model…"):
                    cnn_model, lstm_model = load_models()
                with st.spinner("Detecting deepfake…"):
                    img_arr = np.array(image)
                    time.sleep(0.3)  # small UX pause
                    prob, label = predict_image(cnn_model, img_arr)

                    # Apply user threshold
                    if prob >= threshold:
                        label = "FAKE"
                    else:
                        label = "REAL"

                is_fake = label == "FAKE"
                css_cls = "result-fake" if is_fake else "result-real"
                emoji   = "⚠️" if is_fake else "✅"
                conf    = prob if is_fake else 1 - prob

                st.markdown(f"""
                <div class="{css_cls}">
                    <div class="result-label">{emoji} {label}</div>
                    <div class="confidence-text">Confidence: <strong>{conf*100:.1f}%</strong></div>
                    <br>
                    <span class="model-badge">EfficientNet B0 CNN</span>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("**Raw fake probability:**")
                st.progress(min(prob, 1.0))
                st.caption(f"{prob*100:.2f}% probability of being a deepfake")

                if is_fake:
                    st.markdown("""
                    <div class="info-box">
                        ⚠️ This image shows signs of AI-generated or manipulated facial content.
                        Typical artifacts include unnatural skin texture, eye asymmetry, or blurred edges.
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="info-box">
                        ✅ No significant deepfake artifacts were detected. The image appears authentic.
                    </div>
                    """, unsafe_allow_html=True)

# ════════════════════ VIDEO TAB ═══════════════════════════════════════════════
with tab_vid:
    st.markdown("#### Upload a video to check for temporal deepfake patterns")
    uploaded_vid = st.file_uploader(
        "Choose video", type=["mp4","avi","mov","mkv"],
        key="vid_upload", label_visibility="collapsed"
    )

    if uploaded_vid:
        st.video(uploaded_vid)

        if st.button("🔍 Analyze Video (CNN + LSTM)", key="btn_vid"):
            with st.spinner("Loading models…"):
                cnn_model, lstm_model = load_models()

            progress_bar = st.progress(0, text="Extracting frames…")

            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(uploaded_vid.read())
                tmp_path = tmp.name

            progress_bar.progress(20, text="Running CNN feature extraction…")
            time.sleep(0.2)

            try:
                prob, label, frames_bgr = predict_video(
                    cnn_model, lstm_model, tmp_path,
                    n_frames=n_frames
                )
            finally:
                os.unlink(tmp_path)

            progress_bar.progress(80, text="Running BiLSTM temporal analysis…")
            time.sleep(0.2)
            progress_bar.progress(100, text="Done!")

            # Apply user threshold
            label = "FAKE" if prob >= threshold else "REAL"

            is_fake = label == "FAKE"
            css_cls = "result-fake" if is_fake else "result-real"
            emoji   = "⚠️" if is_fake else "✅"
            conf    = prob if is_fake else 1 - prob

            st.markdown(f"""
            <div class="{css_cls}">
                <div class="result-label">{emoji} {label}</div>
                <div class="confidence-text">Confidence: <strong>{conf*100:.1f}%</strong></div>
                <br>
                <span class="model-badge">EfficientNet B0 + BiLSTM</span>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**Raw fake probability across sequence:**")
            st.progress(min(prob, 1.0))
            st.caption(f"{prob*100:.2f}% probability of being a deepfake video")

            # Show sampled frames
            if frames_bgr:
                st.markdown("#### 📽️ Sampled Frames Used for Analysis")
                display_frames = frames_bgr[:min(8, len(frames_bgr))]
                cols = st.columns(len(display_frames))
                for idx, (col, fr) in enumerate(zip(cols, display_frames)):
                    rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
                    col.image(rgb, caption=f"Frame {idx+1}", use_column_width=True)

            if is_fake:
                st.markdown("""
                <div class="info-box">
                    ⚠️ The temporal analysis of facial movements across frames reveals
                    inconsistencies consistent with AI-generated deepfake content.
                    The BiLSTM model detected unnatural motion patterns or feature drift.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="info-box">
                    ✅ The temporal sequence of facial features appears natural and consistent.
                    No deepfake artifacts were detected across the analyzed frames.
                </div>
                """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#64748b; font-size:0.85rem; padding:1rem 0;">
    DeepFake Detector · EfficientNet B0 + BiLSTM · For educational & research use only
</div>
""", unsafe_allow_html=True)
