import streamlit as st
import cv2
import tempfile
import os
from ultralytics import YOLO

st.set_page_config(page_title="AI Surveillance System", layout="wide")

st.title(" AI Surveillance System - Weapon Detection")

# Load model once — falls back to yolov8n.pt if best.pt is missing
@st.cache_resource
def load_model():
    # Try custom trained model first
    for model_path in ["models/best.pt", "yolo26n.pt", "yolov8n.pt"]:
        if os.path.exists(model_path):
            st.info(f"Loaded model: `{model_path}`")
            return YOLO(model_path)

    # If nothing found locally, download yolov8n from ultralytics (always works)
    st.warning("⚠️ No local model found. Using default YOLOv8n (not trained for weapons).")
    return YOLO("yolov8n.pt")  # ultralytics auto-downloads this

model = load_model()

option = st.sidebar.selectbox(
    "Select Detection Mode",
    ("Image Upload", "Video Upload", "Webcam")
)

# ---------------- IMAGE DETECTION ----------------
if option == "Image Upload":

    st.header("📷 Upload Image")
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_path = temp_file.name

        with st.spinner("Running detection..."):
            results = model(temp_path)
            result_image = results[0].plot()

        st.image(result_image, channels="BGR", caption="Detection Result")
        os.remove(temp_path)


# ---------------- VIDEO DETECTION ----------------
elif option == "Video Upload":

    st.header("🎥 Upload Video")
    uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

    if uploaded_video is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            temp_video.write(uploaded_video.read())
            video_path = temp_video.name

        cap = cv2.VideoCapture(video_path)
        stframe = st.empty()
        stop_btn = st.button("⏹ Stop")

        while cap.isOpened() and not stop_btn:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)
            frame = results[0].plot()
            stframe.image(frame, channels="BGR")

        cap.release()
        os.remove(video_path)
        st.success(" Video processing complete.")


# ---------------- WEBCAM ----------------
elif option == "Webcam":

    st.header(" Webcam Detection")

    # Detect if running on Streamlit Cloud (no /dev/video0)
    is_cloud = not os.path.exists("/dev/video0")

    if is_cloud:
        st.error(
            st.info(
    "📹 **Webcam detection is a computer vision feature that works locally.**\n\n"
    "This feature uses real-time video stream processing with OpenCV and YOLOv8. "
    "Live webcam inference will be available in a future update of this app.\n\n"
    "To try it now on your local machine:\n"
    "```\nstreamlit run app/streamlit_app.py\n```"
)
        )
    else:
        run = st.checkbox("▶️ Start Webcam")

        if run:
            camera = cv2.VideoCapture(0)
            FRAME_WINDOW = st.image([])

            while run:
                ret, frame = camera.read()
                if not ret:
                    st.warning("Camera not detected.")
                    break

                results = model(frame)
                frame = results[0].plot()
                FRAME_WINDOW.image(frame, channels="BGR")

            camera.release()