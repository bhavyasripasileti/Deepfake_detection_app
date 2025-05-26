import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
import tempfile
import os

# Function to extract frames from the video
def extract_frames_from_video(video_path, max_frames=30):
    """Extract frames from a video file."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, total_frames // max_frames)

    count = 0
    while len(frames) < max_frames and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % interval == 0:
            frame = cv2.resize(frame, (128, 128)) / 255.0  # Normalize and resize
            if frame.shape[-1] == 4:  # If it has an alpha channel
                frame = frame[..., :3]  # Use only RGB channels
            frames.append(frame)
        count += 1

    cap.release()
    return np.array(frames)

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
            frames_input = np.expand_dims(frames, axis=0)

            # Assuming the model expects two inputs, prepare the second input accordingly
            second_input = np.zeros((1, frames_input.shape[1], 1))  # Adjust this based on the expected shape of second input
            try:
                inputs = [frames_input, second_input]  # Update inputs based on your model requirement
                prediction = model.predict(inputs)
                st.write("Fake" if prediction[0][0] > 0.5 else "Real")
            except Exception as e:
                st.error(f"Error during prediction: {e}")

    else:
        # Handle image upload
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        img_array = preprocess_image(image)

        try:
            # Assuming model expects two inputs, prepare the second input accordingly for image
            second_input = np.zeros((1, 1))  # Adjust based on the second input's requirement
            inputs = [img_array, second_input]  # Update inputs based on your model requirement
            prediction = model.predict(inputs)
            st.write("Fake" if prediction[0][0] > 0.5 else "Real")
        except Exception as e:
            st.error(f"Error during prediction: {e}")

