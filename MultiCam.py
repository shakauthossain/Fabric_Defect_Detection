import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from ultralytics import YOLO
import cv2
import numpy as np
import requests

# Load the trained YOLOv8 model
model = YOLO('best.pt')  # Replace 'best.pt' with your YOLOv8 model path

# Streamlit App Configuration
st.set_page_config(
    page_title="Multi-Camera Object Detection",
    page_icon="\ud83d\udcfc",
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
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.6, 0.01)

camera_options = ["Webcam (Default)", "External Camera (Index 1)", "ESP32-CAM (HTTP URL)", "RTSP Stream"]
camera_source = st.sidebar.selectbox("Select Camera Source", camera_options)

if camera_source == "ESP32-CAM (HTTP URL)":
    esp32_url = st.sidebar.text_input("Enter ESP32-CAM Stream URL", "http://192.168.0.113/stream")

elif camera_source == "RTSP Stream":
    rtsp_url = st.sidebar.text_input("Enter RTSP Stream URL", "rtsp://username:password@192.168.0.1:554/stream")

start_stream = st.sidebar.button("Start Streaming")
stop_stream = st.sidebar.button("Stop Streaming")

# Function to fetch a single frame from ESP32-CAM
def read_esp32_frame(url):
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

# Function to start the video feed
def start_feed(source):
    cap = None

    if source == "Webcam (Default)":
        cap = cv2.VideoCapture(0)

    elif source == "External Camera (Index 1)":
        cap = cv2.VideoCapture(1)

    elif source == "ESP32-CAM (HTTP URL)":
        cap = "ESP32"

    elif source == "RTSP Stream":
        cap = cv2.VideoCapture(rtsp_url)

    if isinstance(cap, cv2.VideoCapture) and not cap.isOpened():
        st.error("Failed to open camera. Check source or connection.")
        cap = None

    return cap

# Function to stop the video feed
def stop_feed(cap):
    if cap is not None and isinstance(cap, cv2.VideoCapture):
        cap.release()
    cv2.destroyAllWindows()

# Video Transformer for YOLO Inference
class YOLOVideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = model

    def transform(self, frame):
        # Convert frame to a NumPy array
        img = frame.to_ndarray(format="bgr24")

        # Perform YOLO inference
        results = self.model.predict(source=img, conf=confidence_threshold)

        # Annotate the frame
        annotated_frame = results[0].plot()

        return annotated_frame

# Main Application Logic
if start_stream:
    st.success("Starting stream...")
    cap = start_feed(camera_source)

    while not stop_stream:
        if cap == "ESP32":
            frame = read_esp32_frame(esp32_url)
            if frame is None:
                st.error("Failed to fetch ESP32-CAM frame. Check URL and connection.")
                break
        elif isinstance(cap, cv2.VideoCapture):
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to fetch frame. Check the camera source.")
                break
        else:
            st.error("Camera source is not initialized. Exiting stream.")
            break

        # Perform YOLO inference
        results = model.predict(source=frame, conf=confidence_threshold)

        # Annotate the frame
        annotated_frame = results[0].plot()

        # Convert BGR to RGB for Streamlit
        annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        # Display the video feed
        st.image(annotated_frame_rgb, channels="RGB", use_column_width=True)

    # Stop the feed when 'Stop Streaming' is clicked
    if isinstance(cap, cv2.VideoCapture):
        stop_feed(cap)

# Start WebRTC Streamer
if camera_source == "Webcam (Default)":
    webrtc_streamer(
        key="object-detection",
        video_transformer_factory=YOLOVideoTransformer,
        media_stream_constraints={"video": True, "audio": False},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    )

# Footer
st.markdown("<div class='footer'>Built with \ud83d\udca1 using Streamlit and WebRTC | Powered by YOLOv8</div>", unsafe_allow_html=True)
