import streamlit as st
from ultralytics import YOLO
import cv2
import requests
import numpy as np

# Load the trained YOLOv8 model
model = YOLO('best.pt')  # Replace 'best.pt' with the path to your model

# ESP32-CAM stream URL
stream_url = "http://192.168.0.113/stream"

# Streamlit App
st.title("Fabric Defect Detection System")
st.subheader("Fabric Defect Detection System for Garments QC")

# Sidebar Configuration
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.6, 0.01)

# Start Streaming Button
start_stream = st.button("Start System Camera")

# Run video streaming only if Start Stream button is clicked
if start_stream:
    st.success("Camera Has Initialized. Press 'Stop Streaming' to quit.")
    stop_stream = st.button("Stop Streaming")

    # Placeholder for video feed
    video_placeholder = st.empty()

    while not stop_stream:
        # Fetch a single frame from ESP32-CAM stream
        try:
            response = requests.get(stream_url, stream=True, timeout=5)
            bytes_data = bytes()

            for chunk in response.iter_content(chunk_size=1024):
                bytes_data += chunk
                # Check for start and end of the image
                start = bytes_data.find(b'\xff\xd8')
                end = bytes_data.find(b'\xff\xd9')
                if start != -1 and end != -1:
                    # Extract the image bytes
                    jpg_data = bytes_data[start:end + 2]
                    bytes_data = bytes_data[end + 2:]

                    # Decode image
                    frame = cv2.imdecode(np.frombuffer(jpg_data, np.uint8), cv2.IMREAD_COLOR)

                    # Perform YOLO inference
                    results = model.predict(source=frame, conf=confidence_threshold)

                    # Annotate the frame with predictions
                    annotated_frame = results[0].plot()

                    # Convert BGR to RGB for Streamlit
                    annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

                    # Display the annotated frame
                    video_placeholder.image(annotated_frame_rgb, channels="RGB", use_column_width=True)

                    break  # Process only one frame at a time
        except requests.exceptions.RequestException as e:
            st.error(f"Error: Unable to fetch video stream from System Camera. {e}")
            break

        # Stop button control
        # stop_stream = st.button("Stop Streaming")