import streamlit as st
from ultralytics import YOLO
import cv2
import requests
import numpy as np

# Load the trained YOLOv8 model
model = YOLO('best.pt')  # Replace 'best.pt' with the path to your model

# ESP32-CAM stream URL
stream_url = "http://192.168.0.113/stream"

# Streamlit App Design
st.set_page_config(
    page_title="ESP32-CAM Inference",
    page_icon="📷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a modern and attractive design
st.markdown("""
    <style>
    body {
        background: linear-gradient(to bottom right, #4CAF50, #8BC34A);
        font-family: 'Helvetica', sans-serif;
        color: white;
    }
    .main {
        background-color: rgba(0, 0, 0, 0.6);
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
    }
    .header {
        background-color: rgba(0, 0, 0, 0.7);
        color: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        font-size: 28px;
        font-weight: bold;
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.3);
    }
    .sidebar .sidebar-content {
        background-color: #f9f9f9;
    }
    .button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 15px 30px;
        text-align: center;
        font-size: 18px;
        border-radius: 50px;
        cursor: pointer;
        transition: background-color 0.3s ease;
    }
    .button:hover {
        background-color: #45a049;
    }
    .card {
        background-color: rgba(0, 0, 0, 0.7);
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.3);
        margin-bottom: 20px;
    }
    .video-container {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Header Section
st.markdown("<div class='header'>ESP32-CAM Real-Time Inference with YOLOv8</div>", unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.title("Configuration")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.6, 0.01)

st.sidebar.markdown("#### Stream Control")
start_stream = st.sidebar.button("Start Stream", key="start", help="Click to start streaming the video feed.")
stop_stream = st.sidebar.button("Stop Stream", key="stop", help="Click to stop the video feed.")

# Main content layout using columns
col1, col2 = st.columns([3, 1])

with col1:
    st.markdown("### Live Video Feed", unsafe_allow_html=True)
    video_placeholder = st.empty()

with col2:
    st.markdown("### Controls", unsafe_allow_html=True)
    st.markdown("""
        📍 **Adjust Confidence**: Use the slider to set the confidence threshold for object detection.
        - **Start Stream**: Begin real-time inference.
        - **Stop Stream**: Halt the feed gracefully.
    """, unsafe_allow_html=True)

    # Add an informative section
    st.markdown("""
    #### How it works:
    - The video stream from ESP32-CAM is processed using YOLOv8 for object detection.
    - Detected objects are highlighted with bounding boxes and confidence scores.
    - You can adjust the confidence threshold from the sidebar for accurate detection.
    """, unsafe_allow_html=True)

# Start Streaming
if start_stream:
    st.success("Starting ESP32-CAM stream...")
    try:
        while not stop_stream:
            # Fetch frame from ESP32-CAM
            response = requests.get(stream_url, stream=True, timeout=5)
            bytes_data = bytes()
            for chunk in response.iter_content(chunk_size=1024):
                bytes_data += chunk
                start = bytes_data.find(b'\xff\xd8')
                end = bytes_data.find(b'\xff\xd9')
                if start != -1 and end != -1:
                    jpg_data = bytes_data[start:end+2]
                    bytes_data = bytes_data[end+2:]

                    # Decode image
                    frame = cv2.imdecode(np.frombuffer(jpg_data, np.uint8), cv2.IMREAD_COLOR)

                    # Perform YOLO inference
                    results = model.predict(source=frame, conf=confidence_threshold)

                    # Annotate the frame
                    annotated_frame = results[0].plot()

                    # Convert BGR to RGB for display
                    annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

                    # Display the video feed
                    video_placeholder.image(annotated_frame_rgb, channels="RGB", use_column_width=True)

                    break
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching stream: {e}")

# Footer section with credits and links
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    """
    <div style="text-align: center; font-size: 14px; color: #ffffff;">
    Built with 💚 using Streamlit and YOLOv8 | <a href="https://github.com/ultralytics/ultralytics" target="_blank" style="color: #ffffff;">Ultralytics YOLO</a>
    </div>
    """,
    unsafe_allow_html=True
)