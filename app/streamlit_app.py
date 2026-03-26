import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO
model = YOLO("runs/detect/train/weights/best.pt")

st.title("AI Surveillance System - Weapon Detection")

option = st.sidebar.selectbox(
    "Select Detection Type",
    ("Webcam", "Image Upload", "Video Upload")
)


if option == "Image Upload":

    st.header("Upload Image")

    uploaded_file = st.file_uploader("Choose an image", type=["jpg","jpeg","png"])

    if uploaded_file is not None:

        file_bytes = uploaded_file.read()

        with open("temp.jpg","wb") as f:
            f.write(file_bytes)

        results = model("temp.jpg")

        result_image = results[0].plot()

        st.image(result_image, channels="BGR")

elif option == "Video Upload":

    st.header("Upload Video")

    uploaded_video = st.file_uploader("Upload Video", type=["mp4","avi","mov"])

    if uploaded_video is not None:

        temp_video = tempfile.NamedTemporaryFile(delete=False)

        temp_video.write(uploaded_video.read())

        cap = cv2.VideoCapture(temp_video.name)

        stframe = st.empty()

        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:
                break

            results = model(frame)

            frame = results[0].plot()

            stframe.image(frame, channels="BGR")

        cap.release()

elif option == "Webcam":

    st.header("Webcam Detection")

    run = st.checkbox("Start Webcam")

    FRAME_WINDOW = st.image([])

    camera = cv2.VideoCapture(0)

    while run:

        ret, frame = camera.read()

        if not ret:
            st.write("Camera not detected")
            break

        results = model(frame)

        frame = results[0].plot()

        FRAME_WINDOW.image(frame, channels="BGR")

    camera.release()