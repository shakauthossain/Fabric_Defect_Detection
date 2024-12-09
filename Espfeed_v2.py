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

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
        font-family: 'Arial', sans-serif;
    }
    .header {
        background-color: #4CAF50;
        color: white;
        padding: 10px;
        border-radius: 10px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
    }
    .sidebar .sidebar-content {
        background-color: #f9f9f9;
    }
    .button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 20px;
        text-align: center;
        font-size: 16px;
        margin: 4px 2px;
        border-radius: 10px;
        cursor: pointer;
    }
    .button:hover {
        background-color: #45a049;
    }
    </style>
""", unsafe_allow_html=True)

# Header Section
st.markdown("<div class='header'>ESP32-CAM Real-Time Inference with YOLOv8</div>", unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.title("Configuration")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.6, 0.01)
st.sidebar.markdown("#### Stream Control")
start_stream = st.sidebar.button("Start Stream", key="start")
stop_stream = st.sidebar.button("Stop Stream", key="stop")

# Layout for Video Feed
col1, col2 = st.columns([3, 1])

with col1:
    st.markdown("### Live Video Feed")
    video_placeholder = st.empty()

with col2:
    st.markdown("### Controls")
    st.write("👈 Adjust the confidence slider from the sidebar to fine-tune detections.")
    st.markdown("""
        - **Start Stream**: Begin real-time inference.
        - **Stop Stream**: Halt the feed gracefully.
        """)

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

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    """
    <div style="text-align: center; font-size: 12px; color: #888;">
    Built with 💚 using Streamlit and YOLOv8 | <a href="https://github.com/ultralytics/ultralytics" target="_blank">Ultralytics YOLO</a>
    </div>
    """,
    unsafe_allow_html=True
)