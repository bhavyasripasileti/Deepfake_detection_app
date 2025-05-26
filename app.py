import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
import tempfile
import os

# Function to extract frames from the video
def extract_frames_from_video(video_path, max_frames=20):
    """Extract frames from a video file."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while len(frames) < max_frames and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (128, 128)) / 255.0  # Normalize and resize
        if frame.shape[-1] == 4:  # If it has an alpha channel
            frame = frame[..., :3]  # Use only RGB channels
        frames.append(frame)
    
    cap.release()

    # Pad with zeros if fewer than max_frames
    while len(frames) < max_frames:
        frames.append(np.zeros((128, 128, 3)))  # Append zero frames if needed
    
    return np.array(frames)

# Function to use a feature extractor
def feature_extractor(frames):
    # Define a pre-trained model, for example, MobileNetV2
    mobile_net = tf.keras.applications.MobileNetV2(include_top=False, input_shape=(128, 128, 3), pooling='avg')
    features = mobile_net.predict(frames)
    return features  # Shape will be (num_frames, 2048)

# Function to preprocess a single image
def preprocess_image(image):
    """Resize and normalize the image."""
    image = image.resize((128, 128))  # Resize to the required input shape
    img_array = np.array(image) / 255.0  # Normalize to [0, 1]
    if img_array.shape[-1] == 4:  # If it has an alpha channel
        img_array = img_array[..., :3]  # Use only RGB channels
    return np.expand_dims(img_array, axis=0)  # Expand dims to (1, 128, 128, 3)

# Function to load model given a filename
@st.cache_resource
def load_model(model_name):
    return tf.keras.models.load_model(os.path.join("model", model_name))

# Title of the app
st.title("Deepfake Face Detection")

# File uploader for image or video
uploaded_file = st.file_uploader("Upload a face image or video", type=["jpg", "png", "mp4"])

# Function to determine which model to load based on the uploaded file type
def determine_model(uploaded_file):
    if uploaded_file is not None:
        if uploaded_file.type in ["video/mp4"]:
            return "CNN_RNN.h5"  # You can choose any model appropriate for videos
        elif uploaded_file.type in ["image/jpeg", "image/png"]:
            return "new_model.h5"  # Choose a model appropriate for images
    return None

# Automatically select the model based on the uploaded file
selected_model = determine_model(uploaded_file)
model = load_model(selected_model) if selected_model else None

if uploaded_file:
    if uploaded_file.type == "video/mp4":
        st.video(uploaded_file)

        # Save video to a temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        frames = extract_frames_from_video(tfile.name)

        if frames.shape[0] < 10:
            st.warning("Too few frames extracted. Try uploading a longer video.")
        else:
            features = feature_extractor(frames)  # Get features of the frames
            features_input = np.expand_dims(features, axis=0)  # Add batch dimension

            try:
                prediction = model.predict(features_input)
                st.write("Fake" if prediction[0][0] > 0.5 else "Real")
            except Exception as e:
                st.error(f"Error during prediction: {e}")

    else:
        # Handle image upload
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        img_array = preprocess_image(image)

        try:
            prediction = model.predict(img_array)
            st.write("Fake" if prediction[0][0] > 0.5 else "Real")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
