import streamlit as st
import tempfile
import cv2
import numpy as np
from PIL import Image
import os
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

# Extract frames from video
def extract_frames_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (224, 224))
            frames.append(frame)
    finally:
        cap.release()  # Ensure resources are released

    return np.array(frames)

# Extract features using pretrained ResNet50
def extract_features(frames):
    model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    preprocessed = preprocess_input(frames)
    features = model.predict(preprocessed, verbose=0)
    return features

# Cache the loaded model
@st.cache_resource
def load_model(model_path):
    return tf.keras.models.load_model(model_path)

# Display real/fake result
def display_result(prediction):
    if prediction > 0.5:
        st.markdown("<h2 style='color:red;'>👎 Fake</h2>", unsafe_allow_html=True)
    else:
        st.markdown("<h2 style='color:green;'>👍 Real</h2>", unsafe_allow_html=True)

# Main Streamlit app
def main():
    st.title("Deepfake Detection App")

    uploaded_file = st.file_uploader("Upload a video or an image", type=['mp4', 'avi', 'mov', 'jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        tfile.close()  # Close the file after writing

        try:
            model = load_model("model/fixed_model.h5")

            # ---------- VIDEO ----------
            if uploaded_file.name.endswith(('.mp4', '.avi', '.mov')):
                st.video(tfile.name)
                frames = extract_frames_from_video(tfile.name)

                if frames.shape[0] < 20:
                    st.warning("Too few frames extracted. Upload a longer video.")
                    return

                features = extract_features(frames)
                features = features[:20]  # Use first 20 frames
                features = np.expand_dims(features, axis=0)  # Shape: (1, 20, 2048)

                prediction = model.predict(features)[0][0]
                display_result(prediction)

            # ---------- IMAGE ----------
            elif uploaded_file.name.endswith(('.jpg', '.jpeg', '.png')):
                image = Image.open(tfile.name).convert("RGB")
                st.image(image, caption="Uploaded Image", use_column_width=True)

                image = image.resize((224, 224))
                img_array = np.array(image)[:, :, :3]
                img_array = np.expand_dims(img_array, axis=0)
                img_array = preprocess_input(img_array)

                feature_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
                features = feature_model.predict(img_array)
                features = np.tile(features, (20, 1))  # Make into 20-frame sequence
                features = np.expand_dims(features, axis=0)

                prediction = model.predict(features)[0][0]
                display_result(prediction)
        
        finally:
            os.remove(tfile.name)  # Safely delete the temp file

if __name__ == "__main__":
    main()
