import streamlit as st
import cv2
import numpy as np
import requests
import time

st.set_page_config(page_title="♻️ Waste Detection Live", layout="wide")
st.title("♻️ Live Waste Detection (via FastAPI)")

API_URL = "http://127.0.0.1:8000/detect"

CLASS_NAMES = ['brick', 'concrete', 'foam', 'general_w', 'gypsum_board',
               'pipes', 'plastic', 'stone', 'tile', 'wood']

CLASS_COLORS = {
    'brick': (255, 0, 0),
    'concrete': (128, 128, 128),
    'foam': (255, 255, 0),
    'general_w': (0, 255, 255),
    'gypsum_board': (255, 165, 0),
    'pipes': (0, 0, 255),
    'plastic': (0, 255, 0),
    'stone': (160, 82, 45),
    'tile': (255, 0, 255),
    'wood': (139, 69, 19)
}

run = st.toggle("▶️ Start Live Stream")

col1, col2, col3 = st.columns([1, 2.2, 1])
with col2:
    FRAME_WINDOW = st.image([], use_container_width=False)
fps_text = st.empty()

def draw_boxes(frame, boxes, scores, classes):
    for (x1, y1, x2, y2), s, c in zip(boxes, scores, classes):
        label = CLASS_NAMES[c] if c < len(CLASS_NAMES) else f"cls{c}"
        color = CLASS_COLORS.get(label, (0, 255, 0))
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(frame, f"{label} {s:.2f}", (int(x1), int(y1) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame

if run:
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        st.error("❌ Camera is not available.")
    else:
        st.success("✅ Camera is running.")

    while run:
        start = time.time()
        ret, frame = cap.read()
        if not ret:
            st.warning("⚠️ Cannot read from camera.")
            break

        _, img_encoded = cv2.imencode('.jpg', frame)
        files = {"file": ("frame.jpg", img_encoded.tobytes(), "image/jpeg")}

        try:
            r = requests.post(API_URL, files=files, timeout=5)
            if r.status_code == 200:
                data = r.json()
                boxes = data.get("boxes", [])
                scores = data.get("scores", [])
                classes = data.get("classes", [])
                if boxes:
                    frame = draw_boxes(frame, boxes, scores, classes)
            else:
                st.warning(f"⚠️ API Error: {r.status_code}")
        except Exception as e:
            st.warning(f"⚠️ API Connection Error: {e}")

        resized_frame = cv2.resize(frame, (960, int(frame.shape[0] * (960 / frame.shape[1]))))
        FRAME_WINDOW.image(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB), use_container_width=False)

        fps = 1 / (time.time() - start)
        fps_text.markdown(f"**⚡ FPS:** {fps:.1f}")

        time.sleep(0.02)

    cap.release()
else:
    st.info("🎥 Click the toggle above to start the camera.")
