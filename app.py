import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import os

# Load the trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("deepfake_model.h5")  # adjust path if needed
    return model

model = load_model()

# Preprocess image
def preprocess_image(img):
    img = img.convert("RGB")
    img = img.resize((224, 224))  # adjust if your model expects a different size
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# App layout
st.title("Deepfake Detection Web App")
st.write("Upload an image frame from a video to detect if it's fake or real.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_array = preprocess_image(image)

    # Create dummy second input (adjust shape as per model)
    dummy_input = np.zeros((1, 10))  # Change shape if your model expects something else

    try:
        prediction = model.predict([img_array, dummy_input])[0][0]
        if prediction > 0.5:
            st.error(f"Prediction: Fake ({prediction:.2f})")
        else:
            st.success(f"Prediction: Real ({1 - prediction:.2f})")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
