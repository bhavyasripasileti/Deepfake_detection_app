import streamlit as st
import tempfile
import cv2
import numpy as np
from PIL import Image
import os
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

# Function to extract frames from the uploaded video
def extract_frames_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (224, 224))  # Resize frames to (224, 224)
        frames.append(frame)
    
    cap.release()
    return np.array(frames)

# Function to extract features from frames using a pre-trained model
def extract_features(frames):
    model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    features = model.predict(preprocess_input(frames))
    return features

# Load the model
@st.cache_resource
def load_model(model_path):
    return tf.keras.models.load_model(model_path)

# Main app function
def main():
    st.title("Deepfake Detection App")

    uploaded_file = st.file_uploader("Upload a video or an image", type=['mp4', 'avi', 'mov', 'jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        # Save uploaded file to a temporary location
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        # ----- VIDEO HANDLING -----
        if uploaded_file.name.endswith(('.mp4', '.avi', '.mov')):
            st.video(tfile.name)  # Show the video in Streamlit

            frames = extract_frames_from_video(tfile.name)

            if frames.shape[0] < 10:
                st.warning("Too few frames extracted. Try uploading a longer video.")
            else:
                st.success(f"{len(frames)} frames extracted from video.")

                features = extract_features(frames)

                if features.shape[0] >= 20:
                    features = features[:20]
                else:
                    st.warning("Not enough frames (20 required) after feature extraction.")
                    return

                features = np.expand_dims(features, axis=0)  # Shape: (1, 20, 2048)

                context_input = np.random.random((1, 20))

                model = load_model("model/CNN_RNN.h5")

                prediction = model.predict([features, context_input])

                if prediction[0][0] > 0.5:
                    st.markdown("<h2 style='color:red;'>👎 Fake</h2>", unsafe_allow_html=True)
                else:
                    st.markdown("<h2 style='color:green;'>👍 Real</h2>", unsafe_allow_html=True)

        # ----- IMAGE HANDLING -----
        elif uploaded_file.name.endswith(('.jpg', '.jpeg', '.png')):
            image = Image.open(tfile.name).convert("RGB")
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # Resize and preprocess
            image = image.resize((224, 224))
            img_array = np.array(image)[:, :, :3]
            img_array = np.expand_dims(img_array, axis=0)  # (1, 224, 224, 3)
            img_array = preprocess_input(img_array)

            # Extract features from image
            feature_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
            features = feature_model.predict(img_array)  # (1, 2048)

            # Create a fake 20-frame sequence by tiling the image features
            features = np.tile(features, (20, 1))  # (20, 2048)
            features = np.expand_dims(features, axis=0)  # (1, 20, 2048)

            context_input = np.random.random((1, 20))

            model = load_model("model/new_model.h5")

            prediction = model.predict([features, context_input])

            if prediction[0][0] > 0.5:
                st.markdown("<h2 style='color:red;'>👎 Fake</h2>", unsafe_allow_html=True)
            else:
                st.markdown("<h2 style='color:green;'>👍 Real</h2>", unsafe_allow_html=True)

        # Clean up temporary file
        os.remove(tfile.name)

if __name__ == "__main__":
    main()
