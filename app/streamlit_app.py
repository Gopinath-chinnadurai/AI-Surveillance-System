import streamlit as st
import cv2
import tempfile
import os
from ultralytics import YOLO

st.set_page_config(page_title="AI Surveillance System", layout="wide")

st.title(" AI Surveillance System - Weapon Detection")

# Load model once
@st.cache_resource
def load_model():
    model_path = "models/best.pt"

    if not os.path.exists(model_path):
        st.error("Model file not found. Please upload best.pt in models folder.")
        st.stop()

    return YOLO(model_path)

model = load_model()

option = st.sidebar.selectbox(
    "Select Detection Type",
    ("Image Upload", "Video Upload", "Webcam")
)

# ---------------- IMAGE DETECTION ----------------
if option == "Image Upload":

    st.header(" Upload Image")

    uploaded_file = st.file_uploader("Choose an image", type=["jpg","jpeg","png"])

    if uploaded_file is not None:

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_path = temp_file.name

        results = model(temp_path)

        result_image = results[0].plot()

        st.image(result_image, channels="BGR", caption="Detection Result")

        os.remove(temp_path)


# ---------------- VIDEO DETECTION ----------------
elif option == "Video Upload":

    st.header(" Upload Video")

    uploaded_video = st.file_uploader("Upload Video", type=["mp4","avi","mov"])

    if uploaded_video is not None:

        with tempfile.NamedTemporaryFile(delete=False) as temp_video:
            temp_video.write(uploaded_video.read())
            video_path = temp_video.name

        cap = cv2.VideoCapture(video_path)

        stframe = st.empty()

        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:
                break

            results = model(frame)

            frame = results[0].plot()

            stframe.image(frame, channels="BGR")

        cap.release()
        os.remove(video_path)


# ---------------- WEBCAM ----------------
elif option == "Webcam":

    st.header(" Webcam Detection")

    st.warning("⚠ Webcam works only when running locally, not on Streamlit Cloud.")

    run = st.checkbox("Start Webcam")

    if run:

        camera = cv2.VideoCapture(0)
        FRAME_WINDOW = st.image([])

        while run:

            ret, frame = camera.read()

            if not ret:
                st.write("Camera not detected")
                break

            results = model(frame)

            frame = results[0].plot()

            FRAME_WINDOW.image(frame, channels="BGR")

        camera.release()