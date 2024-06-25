import streamlit as st
import torch
import cv2
import numpy as np

# Memuat model YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)

# Fungsi untuk mendeteksi bahasa isyarat
def detect_sign_language(frame, conf_threshold):
    # Set threshold ke parameter model
    model.conf = conf_threshold
    results = model(frame)
    return results

def main():
    st.title("Deteksi Bahasa Isyarat Real-Time Menggunakan YOLOv5")
    st.write("Aplikasi ini mendeteksi bahasa isyarat dari video webcam secara real-time menggunakan model YOLOv5.")
    
    # Tambahkan slider untuk mengatur nilai confidence threshold
    conf_threshold = st.slider('Confidence Threshold', 0.0, 1.0, 0.4)

    run = st.checkbox('Mulai Deteksi')

    cap = cv2.VideoCapture(0)  # Buka webcam
    frame_window = st.image([])  # Placeholder untuk frame video

    while run:
        ret, frame = cap.read()
        if not ret:
            st.write("Gagal membuka kamera.")
            break

        # Mengubah frame ke RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Deteksi bahasa isyarat dengan threshold yang dipilih
        results = detect_sign_language(frame_rgb, conf_threshold)
        
        # Menggambar hasil deteksi di frame
        for *xyxy, conf, cls in results.xyxy[0]:
            label = f'{model.names[int(cls)]} {conf:.2f}'
            cv2.rectangle(frame_rgb, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
            cv2.putText(frame_rgb, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Update placeholder dengan frame yang baru
        frame_window.image(frame_rgb, channels="RGB")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
