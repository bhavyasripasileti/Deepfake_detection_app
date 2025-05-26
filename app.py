import streamlit as st
import tempfile
import cv2
import numpy as np
from PIL import Image
import os
import tensorflow as tf

# Function to extract frames from the uploaded video
def extract_frames_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Resize and normalize frames
        frame = cv2.resize(frame, (224, 224)) / 255.0  
        frames.append(frame)
    
    cap.release()
    return np.array(frames)

# Function to preprocess the image
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize image to the required size
    img_array = np.array(image) / 255.0  # Normalize the image
    return img_array

# Load the model
@st.cache_resource
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
        
        if uploaded_file.name.endswith(('.mp4', '.avi', '.mov')):
            # Extract frames from video
            frames = extract_frames_from_video(tfile.name)
            if frames.shape[0] < 10:
                st.warning("Too few frames extracted. Try uploading a longer video.")
            else:
                st.success(f"{len(frames)} frames extracted from video.")
                # Load the model for video prediction
                model = load_model("model/CNN_RNN.h5")  # Ensure this model is present in your "model" directory
                
                # Prepare the input for the model
                frames_input = np.expand_dims(frames, axis=0)  # Adjust input shape based on model requirement

                # Assuming the model requires a second input; for example, this can be a metadata or sequence length
                # Here I'm using a placeholder for the second input, adapt accordingly
                additional_input = np.array([frames.shape[0]])  # Example, replace with your actual second input data
                additional_input = np.expand_dims(additional_input, axis=0)  # Make it batch shape compatible

                # Placeholder for model prediction
                prediction = model.predict([frames_input, additional_input])  # Change to pass both inputs
                st.write("Fake" if prediction[0][0] > 0.5 else "Real")  # Example condition based on model prediction

        elif uploaded_file.name.endswith(('.jpg', '.jpeg', '.png')):
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            img_array = preprocess_image(image)
            img_array = np.expand_dims(img_array, axis=0)  # Prepare the image for model input
            
            # Load the specific model for image prediction
            model = load_model("model/new_model.h5")  # Ensure this model is present in your "model" directory
            
            # Placeholder for model prediction
            prediction = model.predict(img_array)  # This assumes model for images only requires a single input
            st.write("Fake" if prediction[0][0] > 0.5 else "Real")  # Example condition based on prediction

        # Clean up the temporary file after processing
        os.remove(tfile.name)

if __name__ == "__main__":
    main()
