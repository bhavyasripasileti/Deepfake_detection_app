import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
import tempfile

# Load the model once with caching
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/new_model.h5")

model = load_model()

st.title("Deepfake Face Detection")

uploaded_file = st.file_uploader("Upload a face image or video", type=["jpg", "png", "mp4"])

def preprocess_image(image):
    image = image.resize((128, 128))
    return np.array(image) / 255.0

def extract_frames_from_video(video_path, max_frames=30):
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
            frame = cv2.resize(frame, (128, 128))
            frame = frame / 255.0
            frames.append(frame)
        count += 1

    cap.release()
    frames = np.array(frames)
    return frames

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
            frames_input = np.expand_dims(frames, axis=0)  # Shape: (1, num_frames, 128, 128, 3)
            prediction = model.predict(frames_input)
            st.write("Fake" if prediction[0][0] > 0.5 else "Real")

    else:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        img_array = preprocess_image(image)
        prediction = model.predict(np.expand_dims(img_array, axis=0))
        st.write("Fake" if prediction[0][0] > 0.5 else "Real")


