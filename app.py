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
    model = ResNet50(weights='imagenet', include_top=False, pooling='avg')  # Create the model
    features = model.predict(preprocess_input(frames))  # Use the model to extract features
    return features

# Load the model
@st.cache(allow_output_mutation=True)
def load_model(model_path):
    return tf.keras.models.load_model(model_path)

# Function for the main app workflow
def main():
    st.title("Deepfake Detection App")

    uploaded_file = st.file_uploader("Upload a video or an image", type=['mp4', 'jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        # Save video/image to a temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        try:
            if uploaded_file.name.endswith(('.mp4', '.avi', '.mov')):
                # 👇 Show the uploaded video
                st.video(tfile.name)

                # Extract frames from video
                frames = extract_frames_from_video(tfile.name)
                if frames.shape[0] < 10:
                    st.warning("Too few frames extracted. Try uploading a longer video.")
                else:
                    st.success(f"{len(frames)} frames extracted from video.")
                    # Extract features from frames
                    features = extract_features(frames)  # Shape will be (num_frames, 2048)

                    # Adjusting for RNN input
                    if features.shape[0] >= 20:
                        features = features[:20]
                    else:
                        st.warning("Not enough frames (20 required) after feature extraction.")
                        return

                    features = np.expand_dims(features, axis=0)  # Shape: (1, 20, 2048)

                    # Dummy context input for video
                    context_input = np.random.random((1, 20))

                    # Load the model
                    model = load_model("model/CNN_RNN.h5")

                    # Prediction
                    predictions = model.predict([features, context_input])
                    st.write("Prediction:", "🟥 **Fake**" if predictions[0][0] > 0.5 else "🟩 **Real**")

            elif uploaded_file.name.endswith(('.jpg', '.jpeg', '.png')):
                # Show the uploaded image
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)

                img_array = np.array(image.resize((224, 224)))
                img_array = np.expand_dims(img_array, axis=0)

                # Load the specific model for image prediction
                model = load_model("model/new_model.h5")

                # Predict on the image
                prediction = model.predict(img_array)
                st.write("Prediction:", "🟥 **Fake**" if prediction[0][0] > 0.5 else "🟩 **Real**")

        except Exception as e:
            st.error(f"Error: {e}")

        finally:
            os.remove(tfile.name)

if __name__ == "__main__":
    main()
