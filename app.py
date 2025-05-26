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
        frame = cv2.resize(frame, (224, 224))  # Resize to (224, 224)
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

# Main App
def main():
    st.title("Deepfake Detection App")

    uploaded_file = st.file_uploader("Upload a video or an image", type=['mp4', 'avi', 'mov', 'jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        # Save uploaded file to a temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        try:
            if uploaded_file.name.endswith(('.mp4', '.avi', '.mov')):
                # 🔊 Play uploaded video
                st.video(tfile.name)

                # Extract frames
                frames = extract_frames_from_video(tfile.name)
                if frames.shape[0] < 10:
                    st.warning("Too few frames extracted. Try uploading a longer video.")
                    return
                st.success(f"{len(frames)} frames extracted.")

                # Extract features from first 20 frames
                features = extract_features(frames)
                if features.shape[0] >= 20:
                    features = features[:20]
                else:
                    st.warning("At least 20 frames required after feature extraction.")
                    return

                features = np.expand_dims(features, axis=0)  # (1, 20, 2048)
                context_input = np.random.random((1, 20))   # Dummy context input (1, 20)

                # Load video model and predict
                model = load_model("model/CNN_RNN.h5")
                prediction = model.predict([features, context_input])
                st.write("Prediction:", "🟥 **Fake**" if prediction[0][0] > 0.5 else "🟩 **Real**")

            elif uploaded_file.name.endswith(('.jpg', '.jpeg', '.png')):
                # 🖼️ Show uploaded image
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)

                # Preprocess image
                img_array = np.array(image.resize((224, 224)))[:, :, :3]  # Remove alpha if present
                img_array = np.expand_dims(img_array, axis=0)  # (1, 224, 224, 3)

                # Create dummy context input
                context_input = np.random.random((1, 20))  # Adjust as per your model input

                # Load image model and predict
                model = load_model("model/new_model.h5")
                prediction = model.predict([img_array, context_input])
                st.write("Prediction:", "🟥 **Fake**" if prediction[0][0] > 0.5 else "🟩 **Real**")

        except Exception as e:
            st.error(f"Error: {e}")
        finally:
            os.remove(tfile.name)

if __name__ == "__main__":
    main()
