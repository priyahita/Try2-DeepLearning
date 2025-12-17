import streamlit as st
from ultralytics import YOLO
import cv2
import time
# import pyttsx3
import threading

# ======================
# STREAMLIT CONFIG
# ======================
st.set_page_config(
    page_title="PPE Detection",
    layout="wide"
)

st.title("🦺 Real-Time PPE Detection with Warning System")

# ======================
# LOAD MODEL
# ======================
@st.cache_resource
def load_model():
    return YOLO("model_ppe_tuned.pt")
    

model = load_model()

# ======================
# WARNING TIMER
# ======================
last_warning_time = {
    "NO-Hardhat": 0,
    "NO-Mask": 0,
    "NO-Safety Vest": 0
}
WARNING_INTERVAL = 20

# ======================
# TEXT TO SPEECH
# ======================
# def speak_warning(text):
#     engine = pyttsx3.init()
#     engine.setProperty("rate", 150)
#     engine.say(text)
#     engine.runAndWait()

# ======================
# UI CONTROL
# ======================
run = st.checkbox("▶ Start Camera")

FRAME_WINDOW = st.image([])
status_placeholder = st.empty()

# ======================
# CAMERA
# ======================
cap = cv2.VideoCapture(0)

while run:
    ret, frame = cap.read()
    if not ret:
        st.warning("Camera not detected")
        break

    status = {
        "Hardhat": "No",
        "Mask": "No",
        "Safety Vest": "No"
    }

    small_frame = cv2.resize(frame, (640, 480))
    results = model(small_frame, verbose=False)

    missing_items = set()

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            color = (0, 255, 0) if "NO" not in label else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
            )

            if label == "Hardhat":
                status["Hardhat"] = "Yes"
            elif label == "Mask":
                status["Mask"] = "Yes"
            elif label == "Safety Vest":
                status["Safety Vest"] = "Yes"

            if "NO" in label:
                missing_items.add(label)

    # ======================
    # WARNING LOGIC
    # ======================
    current_time = time.time()
    active_warnings = []
    active_apparel = []

    for item in missing_items:
        if current_time - last_warning_time[item] > WARNING_INTERVAL:
            if item == "NO-Hardhat":
                active_warnings.append("Risk of head injury")
                active_apparel.append("Hardhat")
            elif item == "NO-Mask":
                active_warnings.append("Risk of respiratory injury")
                active_apparel.append("Mask")
            elif item == "NO-Safety Vest":
                active_warnings.append("Risk of low visibility")
                active_apparel.append("Safety Vest")
            last_warning_time[item] = current_time

    if active_warnings:
        warning_text = (
            "WARNING: "
            + " and ".join(active_warnings)
            + ". Please wear "
            + " and ".join(active_apparel)
        )

        # threading.Thread(
        #     target=speak_warning,
        #     args=(warning_text,),
        #     daemon=True
        # ).start()

        cv2.putText(
            frame, warning_text, (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
        )

    # ======================
    # STATUS DISPLAY
    # ======================
    status_text = ""
    for k, v in status.items():
        status_text += f"**{k}** : {v}  \n"

    status_placeholder.markdown(status_text)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(frame)

cap.release()
