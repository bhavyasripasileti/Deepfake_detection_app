import streamlit as st
import tempfile
import cv2
import numpy as np
from PIL import Image
import os
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

# Extract frames with skipping and limit
def extract_frames_from_video(video_path, max_frames=100, skip_rate=10):
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or len(frames) >= max_frames:
            break
        if count % skip_rate == 0:
            frame = cv2.resize(frame, (224, 224))
            frames.append(frame)
        count += 1

    cap.release()
    return np.array(frames)

# Extract CNN features from ResNet50
def extract_features(frames):
    model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    frames = preprocess_input(frames)
    features = model.predict(frames, verbose=0)
    return features

# Cache model loading
@st.cache_resource
def load_model(model_path):
    return tf.keras.models.load_model(model_path)

# App starts here
def main():
    st.title("Deepfake Detection App")

    uploaded_file = st.file_uploader("Upload a video or image", type=['mp4', 'avi', 'mov', 'jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        is_video = uploaded_file.name.endswith(('.mp4', '.avi', '.mov'))
        is_image = uploaded_file.name.endswith(('.jpg', '.jpeg', '.png'))

        if is_video:
            st.video(tfile.name)

            with st.spinner("Extracting and processing frames..."):
                frames = extract_frames_from_video(tfile.name)

            st.success(f"{len(frames)} frames extracted (skipped every 10).")

            if len(frames) < 10:
                st.warning("Too few frames to make prediction.")
                os.remove(tfile.name)
                return

            with st.spinner("Extracting features..."):
                features = extract_features(frames)

            # Ensure 20-frame length for model
            if features.shape[0] >= 20:
                features = features[:20]
            else:
                st.warning("Not enough valid frames after processing.")
                os.remove(tfile.name)
                return

            features = np.expand_dims(features, axis=0)  # (1, 20, 2048)
            context_input = np.ones((1, 20))  # Use fixed context input

            model = load_model("model/CNN_RNN.h5")

            with st.spinner("Running prediction..."):
                prediction = model.predict([features, context_input], verbose=0)

            score = prediction[0][0]
            st.write(f"🧪 **Prediction Score:** `{score:.4f}`")

            if score > 0.5:
                st.markdown("<h2 style='color:green;'>👍 Real</h2>", unsafe_allow_html=True)
            else:
                st.markdown("<h2 style='color:red;'>👎 Fake</h2>", unsafe_allow_html=True)

        elif is_image:
            image = Image.open(tfile.name).convert("RGB")
            st.image(image, caption="Uploaded Image", use_column_width=True)

            image = image.resize((224, 224))
            img_array = np.expand_dims(np.array(image)[:, :, :3], axis=0)
            img_array = preprocess_input(img_array)

            with st.spinner("Extracting image features..."):
                feature_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
                features = feature_model.predict(img_array, verbose=0)

            # Fake 20-frame sequence from static image
            features = np.tile(features, (20, 1))
            features = np.expand_dims(features, axis=0)  # (1, 20, 2048)
            context_input = np.ones((1, 20))

            model = load_model("model/new_model.h5")

            with st.spinner("Running prediction..."):
                prediction = model.predict([features, context_input], verbose=0)

            score = prediction[0][0]
            st.write(f"🧪 **Prediction Score:** `{score:.4f}`")

            if score > 0.5:
                st.markdown("<h2 style='color:green;'>👍 Real</h2>", unsafe_allow_html=True)
            else:
                st.markdown("<h2 style='color:red;'>👎 Fake</h2>", unsafe_allow_html=True)

        os.remove(tfile.name)

if __name__ == "__main__":
    main()
