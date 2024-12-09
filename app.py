import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np

# Load the trained YOLOv8 model
model = YOLO('best.pt')  # Replace 'best.pt' with the path to your model

# Streamlit App
st.title("YOLOv8 Webcam Inference")
st.subheader("Real-time object detection using YOLOv8")

# Sidebar Configuration
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.6, 0.01)

# Start Webcam Button
start_webcam = st.button("Start Webcam")

# Run webcam only if Start Webcam button is clicked
if start_webcam:
    # Initialize the webcam
    cap = cv2.VideoCapture(0)

    # Check if the webcam is accessible
    if not cap.isOpened():
        st.error("Error: Unable to access the webcam.")
    else:
        st.success("Webcam started. Press 'Stop Webcam' to quit.")
        stop_webcam = st.button("Stop Webcam")

        # Placeholder for video feed
        video_placeholder = st.empty()

        while not stop_webcam:
            ret, frame = cap.read()
            if not ret:
                st.error("Error: Unable to read from webcam.")
                break

            # Perform inference on the current frame
            results = model.predict(source=frame, conf=confidence_threshold)

            # Annotate the frame with predictions
            annotated_frame = results[0].plot()

            # Convert BGR to RGB for Streamlit
            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

            # Display the annotated frame in the video feed placeholder
            video_placeholder.image(annotated_frame_rgb, channels="RGB", use_column_width=True)

            # Stop button control
            # stop_webcam = st.button("Stop Webcam")

        # Release webcam resource
        cap.release()