import streamlit as st
from ultralytics import YOLO
import cv2
import requests
import numpy as np

# Load the trained YOLOv8 model
model = YOLO('best.pt')  # Replace 'best.pt' with your YOLOv8 model path

# Streamlit App Configuration
st.set_page_config(
    page_title="Multi-Camera Object Detection",
    page_icon="📹",
    layout="wide"
)

# Custom CSS for design
st.markdown("""
    <style>
    body {
        background-color: #f9f9f9;
        font-family: 'Arial', sans-serif;
    }

    .header {
        text-align: center;
        font-size: 36px;
        font-weight: 600;
        color: #3E3E3E;
        margin-bottom: 20px;
    }

    .info-box {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
        font-size: 14px;
        color: #333;
    }

    .footer {
        text-align: center;
        font-size: 14px;
        color: #888;
        margin-top: 20px;
    }

    .footer a {
        color: #007BFF;
        text-decoration: none;
    }

    .footer a:hover {
        text-decoration: underline;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("<div class='header'>Multi-Camera Real-Time Object Detection</div>", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Settings")
st.sidebar.markdown("<div class='info-box'>Select the source and configure options for detection.</div>", unsafe_allow_html=True)

camera_options = ["Webcam (Default)", "External Camera (Index 1)", "ESP32-CAM (HTTP URL)", "RTSP Stream"]
camera_source = st.sidebar.selectbox("Select Camera Source", camera_options)

confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.6, 0.01)

if camera_source == "ESP32-CAM (HTTP URL)":
    esp32_url = st.sidebar.text_input("Enter ESP32-CAM Stream URL", "http://192.168.0.113/stream")

elif camera_source == "RTSP Stream":
    rtsp_url = st.sidebar.text_input("Enter RTSP Stream URL", "rtsp://username:password@192.168.0.1:554/stream")

start_stream = st.sidebar.button("Start Streaming")

# Main layout
video_placeholder = st.empty()
st.sidebar.markdown("<div class='info-box'>Press 'Stop Streaming' to end the feed.</div>", unsafe_allow_html=True)
stop_stream = st.sidebar.button("Stop Streaming")

def read_esp32_frame(url):
    """Fetch a single frame from ESP32-CAM."""
    response = requests.get(url, stream=True, timeout=5)
    bytes_data = bytes()
    for chunk in response.iter_content(chunk_size=1024):
        bytes_data += chunk
        start = bytes_data.find(b'\xff\xd8')
        end = bytes_data.find(b'\xff\xd9')
        if start != -1 and end != -1:
            jpg_data = bytes_data[start:end+2]
            bytes_data = bytes_data[end+2:]
            frame = cv2.imdecode(np.frombuffer(jpg_data, np.uint8), cv2.IMREAD_COLOR)
            return frame
    return None

def start_feed(source):
    """Start the video feed from the selected source."""
    if source == "Webcam (Default)":
        #cap = cv2.VideoCapture(0)
        return st.camera_input("Select Camera")

    elif source == "External Camera (Index 1)":
        cap = cv2.VideoCapture(1)

    elif source == "ESP32-CAM (HTTP URL)":
        cap = None  # We'll fetch frames manually from the URL

    elif source == "RTSP Stream":
        cap = cv2.VideoCapture(rtsp_url)

    return cap

def stop_feed(cap):
    """Stop the video feed."""
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()

if start_stream:
    st.success("Starting stream...")
    cap = start_feed(camera_source)

    while not stop_stream:
        if camera_source == "ESP32-CAM (HTTP URL)":
            frame = read_esp32_frame(esp32_url)
            if frame is None:
                st.error("Failed to fetch ESP32-CAM frame. Check URL and connection.")
                break
        else:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to fetch frame. Check the camera source.")
                break

        # Perform YOLO inference
        results = model.predict(source=frame, conf=confidence_threshold)

        # Annotate the frame
        annotated_frame = results[0].plot()

        # Convert BGR to RGB for Streamlit
        annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        # Display the video feed
        video_placeholder.image(annotated_frame_rgb, channels="RGB", use_column_width=True)

    # Stop the feed when 'Stop Streaming' is clicked
    if cap is not None:
        stop_feed(cap)

# Footer
st.markdown("<div class='footer'>Built with 💡 using Streamlit | Powered by YOLOv8</div>", unsafe_allow_html=True)
