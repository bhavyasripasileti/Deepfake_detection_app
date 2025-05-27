Project Title: Deepfake Face Detection using LSTM

![cover](https://github.com/user-attachments/assets/28d76e81-3fb7-4172-a191-22cfd89a655a)

1. Project Overview

Deepfakes are AI-generated videos in which a person's face or voice is manipulated to make them appear to do or say something they never did. These videos can pose serious threats to privacy, trust, and security. With the increasing realism of such content, detecting deepfakes has become a significant challenge.

This project aims to develop a deepfake detection system that analyzes temporal changes in facial expressions across video frames using a combination of Convolutional Neural Networks (CNN) and Long Short-Term Memory (LSTM) networks. The application is built entirely in Python and features a web-based user interface developed using Streamlit for quick deployment and testing.

2. Objective

To create a robust and scalable deepfake detection model based on LSTM for sequential video frame analysis.

To preprocess real-world video data for facial analysis using frame extraction and face detection techniques.

To provide a lightweight, interactive web application for real-time video-based prediction.

To evaluate model performance using precision, recall, F1-score, and overall accuracy metrics.

3. Technology Stack

Programming Language: Python 3.x

Deep Learning Frameworks: TensorFlow, Keras

Computer Vision: OpenCV, MTCNN

Model Types: CNN (ResNet) + LSTM

Web Deployment: Streamlit

Dataset: Deepfake Detection Challenge (DFDC) Dataset by Meta

--Project Structure:
   deepfake-lstm-detection/
├── app.py
├── model/
│ └── lstm_model.h5
├── utils/
│ └── preprocessing.py
├── sample_videos/
├── images/
│ └── architecture_diagram.png
│ └── prediction_ui.png
├── requirements.txt
└── README.md

4. Data Pipeline

Frame Extraction: Video files are first processed to extract individual frames at uniform intervals.

Face Detection: MTCNN is used to detect and crop faces from each frame.

Normalization: Each face is resized to 128x128 pixels and normalized for consistency.

Sequence Generation: Batches of sequential frames (typically 10-20) are grouped to form a time series input for the LSTM model.

5. Model Architecture

CNN Block: Uses pretrained ResNet layers to extract spatial features from each face image.

LSTM Block: A two-layer LSTM processes the sequential feature vectors to learn temporal inconsistencies across frames.

Output Layer: A Dense layer with a sigmoid activation outputs the probability of the input sequence being a deepfake.

This architecture enables the model to detect unnatural changes or inconsistencies in facial features over time, which are often signs of manipulated content.

6. Deployment and User Interface

The project is deployed using Streamlit, an open-source Python framework for building data apps. The web interface allows users to:

Upload short video clips (preferably 5–10 seconds long).

Preview extracted frames from the video.

Submit the video for processing and receive real/fake prediction.

View confidence scores and inference time.

The UI is optimized for responsiveness and real-time interaction. Backend processing is handled using the pretrained .h5 model and face preprocessing functions.

7. Performance Evaluation

Accuracy: 92%

Precision: 91%

Recall: 90%

F1 Score: 90.5%

The model was trained and validated using a balanced subset of the DFDC dataset. Evaluation was conducted on unseen video samples under varied lighting, compression, and facial movement conditions.

8. Challenges and Solutions

High memory consumption during inference:

Resolved by quantizing the model and limiting the length of input sequences.

Latency when uploading and processing large videos:

Solved by sampling frames and restricting video duration.

Deployment issues with Streamlit dependencies:

Addressed by using a clean requirements.txt and virtual environments.

Underperformance on low-quality or compressed videos:

Fixed by including such videos in the training set and using augmentation techniques.

