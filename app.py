import streamlit as st

st.set_page_config(
    page_title="Deepfake Detector",
    page_icon="🕵️",
    layout="wide",
    initial_sidebar_state="expanded",
)

import os
import time
import tempfile
import numpy as np
from PIL import Image

from config import CFG
from inference import get_model, get_lstm_model, get_cnn_extractor, predict_image, predict_video


def inject_css():
    st.markdown("""
    <style>
    .block-container { padding-top: 1.5rem; }
    .verdict-fake {
        background:#fff0f0; border:2px solid #e74c3c;
        border-radius:12px; padding:1rem 1.5rem; margin-bottom:1rem;
    }
    .verdict-real {
        background:#f0fff4; border:2px solid #27ae60;
        border-radius:12px; padding:1rem 1.5rem; margin-bottom:1rem;
    }
    .verdict-unknown {
        background:#fffbf0; border:2px solid #f39c12;
        border-radius:12px; padding:1rem 1.5rem; margin-bottom:1rem;
    }
    .verdict-fake h2   { color:#c0392b; margin:0; }
    .verdict-real h2   { color:#1e8449; margin:0; }
    .verdict-unknown h2{ color:#d35400; margin:0; }
    .verdict-fake p, .verdict-real p, .verdict-unknown p
                       { margin:0.25rem 0 0; font-size:0.9rem; }
    .score-pill {
        display:inline-block; font-size:1.8rem; font-weight:700;
        padding:0.2rem 1rem; border-radius:10px; margin:0.4rem 0;
    }
    .pill-fake { color:#c0392b; background:#fdecea; }
    .pill-real { color:#1e8449; background:#eafaf1; }
    .frame-tag {
        display:inline-block; font-size:11px; font-weight:600;
        padding:2px 7px; border-radius:8px; margin-top:3px;
    }
    div[data-testid="metric-container"] {
        background:var(--secondary-background-color);
        border-radius:10px; padding:0.6rem 1rem;
        border:1px solid rgba(0,0,0,0.07);
    }
    div[data-testid="stImage"] img { border-radius:8px; }
    </style>
    """, unsafe_allow_html=True)


def sidebar():
    with st.sidebar:
        st.title("🕵️ Deepfake Detector")
        st.caption("EfficientNetB4 + MTCNN + BiLSTM")
        st.divider()

        st.subheader("Settings")
        num_frames  = st.slider("Video frames to analyse", 10, 40, CFG.num_frames, 5)
        aggregation = st.radio("Video aggregation", ["mean", "majority"], index=0)

        st.divider()
        st.subheader("Display")
        show_gallery = st.toggle("Frame gallery", value=True)

        st.divider()
        st.subheader("Model info")
        st.markdown(f"""
        | | |
        |---|---|
        | Backbone | EfficientNetB4 |
        | Input | {CFG.face_size}×{CFG.face_size} px |
        | Face detector | MTCNN |
        | Video model | BiLSTM |
        | Threshold | ≥ {CFG.fake_threshold} → FAKE |
        """)
        st.caption("Label: 1 = FAKE · 0 = REAL")

    return dict(num_frames=num_frames, aggregation=aggregation,
                show_gallery=show_gallery)


def render_verdict(result, media_type):
    label      = result["label"]
    score      = result["score"]
    confidence = result["confidence"]
    model_tag  = result.get("model_tag", "CNN")

    if label == "UNKNOWN":
        st.markdown(
            f"<div class='verdict-unknown'><h2>⚠️ No Face Detected</h2>"
            f"<p>{result.get('error','')}</p></div>",
            unsafe_allow_html=True)
        return

    if label == "FAKE":
        st.markdown(
            "<div class='verdict-fake'><h2>🚨 FAKE — Deepfake Detected</h2>"
            "<p>AI manipulation detected. Signs of GAN or face-swap artifacts.</p></div>",
            unsafe_allow_html=True)
    else:
        st.markdown(
            "<div class='verdict-real'><h2>✅ REAL — Authentic Media</h2>"
            "<p>No manipulation detected. The face appears authentic.</p></div>",
            unsafe_allow_html=True)

    pill = "pill-fake" if label == "FAKE" else "pill-real"
    st.markdown(
        f"<div class='score-pill {pill}'>Fake probability: {score:.1%}</div>",
        unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Fake probability",  f"{score:.1%}")
    c2.metric("Real probability",  f"{1-score:.1%}")
    c3.metric("Confidence",        f"{confidence:.1%}")
    c4.metric("Faces analysed" if media_type == "image" else "Frames", str(result["n_faces"]))
    c5.metric("Model", model_tag)

    st.markdown("**Fake probability**")
    st.progress(float(score))


def render_chart(frame_scores):
    if not frame_scores:
        return
    st.subheader("📊 Per-frame fake probability")
    try:
        import plotly.graph_objects as go
        n = len(frame_scores)
        fig = go.Figure(go.Bar(
            x=list(range(n)),
            y=frame_scores,
            marker_color=["#e74c3c" if s >= CFG.fake_threshold else "#27ae60"
                          for s in frame_scores],
            hovertemplate="Frame %{x}<br>Score: %{y:.3f}<extra></extra>",
        ))
        fig.add_shape(type="line", x0=-0.5, x1=n - 0.5,
                      y0=CFG.fake_threshold, y1=CFG.fake_threshold,
                      line=dict(color="#e74c3c", width=1.5, dash="dash"))
        fig.update_layout(
            xaxis_title="Frame index", yaxis_title="Fake probability",
            yaxis=dict(range=[0, 1]),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=40, r=20, t=10, b=40), height=260, showlegend=False,
        )
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.07)")
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        import pandas as pd
        st.bar_chart(pd.DataFrame({"Score": frame_scores}), height=240)


def render_gallery(result):
    faces, scores = result["faces"], result["frame_scores"]
    if len(faces) == 0:
        return
    st.subheader("🖼️ Frame gallery")
    tab1, tab2 = st.tabs(["Most suspicious", "All frames"])
    COLS = CFG.gallery_cols

    def tag(s):
        c = "#e74c3c" if s >= CFG.fake_threshold else "#27ae60"
        l = "FAKE" if s >= CFG.fake_threshold else "REAL"
        return (f"<div class='frame-tag' style='background:{c}22;"
                f"border:1px solid {c};color:{c}'>{l} {s:.0%}</div>")

    with tab1:
        order = np.argsort(scores)[::-1][:10]
        cols  = st.columns(COLS)
        for i, fi in enumerate(order):
            with cols[i % COLS]:
                st.image(faces[fi], use_column_width=True, caption=f"Frame {fi}")
                st.markdown(tag(scores[fi]), unsafe_allow_html=True)

    with tab2:
        cols = st.columns(COLS)
        for fi, face in enumerate(faces):
            with cols[fi % COLS]:
                st.image(face, use_column_width=True, caption=f"#{fi}")
                st.markdown(tag(scores[fi] if fi < len(scores) else 0.0),
                            unsafe_allow_html=True)


def run_image(uploaded, model, settings):
    image = Image.open(uploaded).convert("RGB")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Uploaded image")
        st.image(image, use_column_width=True)
    with col2:
        st.subheader("Result")
        with st.spinner("Analysing …"):
            t0      = time.time()
            result  = predict_image(image, model)
            elapsed = time.time() - t0
        render_verdict(result, "image")
        st.caption(f"Inference time: {elapsed:.2f} s")
    if result["label"] != "UNKNOWN" and settings["show_gallery"]:
        render_gallery(result)


def run_video(uploaded, model, settings):
    suffix = os.path.splitext(uploaded.name)[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name
    try:
        st.subheader("Uploaded video")
        st.video(tmp_path)

        with st.spinner(
            f"Extracting {settings['num_frames']} frames and analysing with CNN + BiLSTM …"
        ):
            t0      = time.time()
            result  = predict_video(tmp_path, model, settings["num_frames"])
            elapsed = time.time() - t0

        st.subheader("Result")
        if "error" not in result:
            model_tag = result.get("model_tag", "CNN+LSTM")
            st.success(
                f"✅ {result['n_faces']} face crops  |  "
                f"{elapsed:.1f} s  |  Model: {model_tag}"
            )

        render_verdict(result, "video")

        if result["label"] != "UNKNOWN":
            render_chart(result["frame_scores"])
            if settings["show_gallery"]:
                render_gallery(result)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def main():
    inject_css()
    settings = sidebar()

    st.markdown("<h1 style='margin-bottom:0'>🕵️ Deepfake Face Detection</h1>",
                unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#888;margin-top:0'>"
        "Upload an image or video to detect AI-manipulated faces.</p>",
        unsafe_allow_html=True)
    st.divider()

    try:
        model = get_model()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

    try:
        get_cnn_extractor()
        get_lstm_model()
    except Exception:
        pass

    uploaded = st.file_uploader(
        "Upload image or video",
        type=["jpg", "jpeg", "png", "mp4", "avi", "mov", "mkv"],
        label_visibility="collapsed",
    )

    if uploaded is None:
        st.info("👆 Upload an image or video to get started.")
        with st.expander("How it works"):
            st.markdown("""
1. **MTCNN** detects and crops the face from each frame.
2. **EfficientNetB4** extracts deep features from the face crop.
3. For **images** — a Dense head outputs the fake probability directly.
4. For **videos** — frame features are passed as a sequence into a **BiLSTM**,
   which analyses temporal inconsistencies across frames before giving a verdict.
5. Per-frame CNN scores are shown in the bar chart for full transparency.
            """)
        return

    if uploaded.name.lower().endswith((".jpg", ".jpeg", ".png")):
        run_image(uploaded, model, settings)
    elif uploaded.name.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        run_video(uploaded, model, settings)
    else:
        st.error("Unsupported file type.")


if __name__ == "__main__":
    main()
