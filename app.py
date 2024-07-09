import streamlit as st
import torch
import cv2
import numpy as np
import pathlib
from pathlib import Path
pathlib.PosixPath = pathlib.WindowsPath

# Load the YOLOv5 model from a local file
model_path = 'C:/Users/Berliani Risqi/Documents/Sign Language Using YOLOV5/bestv2.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)

# Function to detect sign language
def detect_sign_language(frame, conf_threshold=0.1):
    # Set confidence threshold for the model
    model.conf = conf_threshold
    results = model(frame)
    return results

def main():
    st.title("SignSense: Real-Time Sign Language Detection using YOLOv5")
    st.write("Aplikasi ini mendeteksi bahasa isyarat secara real-time menggunakan model YOLOv5.")

    # Initialize session state for detection status
    if 'run_detection' not in st.session_state:
        st.session_state.run_detection = False

    # Toggle button to start and stop detection
    if st.button('Mulai/Berhenti'):
        st.session_state.run_detection = not st.session_state.run_detection

    # Setting up camera
    cap = cv2.VideoCapture(0)
    frame_window = st.image([])  # Placeholder for video frame

    while st.session_state.run_detection:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to open the camera.")
            break

        # Convert frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect sign language with a threshold of 0.8
        results = detect_sign_language(frame_rgb, conf_threshold=0.8)
        
        # Draw detection results on the frame
        for *xyxy, conf, cls in results.xyxy[0]:
            label = f'{model.names[int(cls)]} {conf:.2f}'
            cv2.rectangle(frame_rgb, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
            cv2.putText(frame_rgb, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Update the Streamlit image placeholder with the new frame
        frame_window.image(frame_rgb, channels="RGB")

    # Release the camera when detection is stopped
    # cap.release()
    # cv2.destroyAllWindows()

if __name__ == '__main__':
    main()