# AI Surveillance System

This project implements an **AI-based Surveillance System** that detects suspicious weapons like **Knife and Guns** using computer vision.

The system supports **Webcam, Image Upload, and Video Upload detection** through an interactive **Streamlit dashboard**.

The application uses **Deep Learning and OpenCV** to process visual data and identify potential anomalies.

<img width="1903" height="975" alt="ai_survey" src="https://github.com/user-attachments/assets/a65cf80f-666b-4e0a-80cf-01586140c86d" />


---

## Dataset

For training the detection model, you need a dataset containing images of **weapons such as knives and guns**.

You can download suitable datasets from platforms like:

- **Kaggle**
- **Roboflow**
- **Open Images Dataset**
- Other public computer vision datasets

Choose any dataset that contains labeled images for **weapon detection (Knife, Gun, etc.)** according to your convenience.

After downloading the dataset, place it in the project directory and use it to train the model.

---

## Features

The system currently supports the following capabilities:

- **Webcam Monitoring** – Detect activities in real-time from webcam feed  
- **Image Upload Detection** – Upload an image and analyze it  
- **Video Upload Analysis** – Process uploaded videos frame by frame  
- **AI-based Detection** – Uses deep learning for activity analysis  
- **Streamlit Interface** – Easy-to-use web dashboard  

---

## Project Architecture

```
AI_Surveillance_System
│
├── app
│   └── streamlit_app.py          # Streamlit UI (webcam, image, video upload)
│
├── dataset
│   ├── train
│   │   ├── images
│   │   └── labels
│   │
│   ├── valid
│   │   ├── images
│   │   └── labels
│   │
│   └── data.yaml                 # YOLO dataset configuration
│
├── models
│   └── best.pt                   # Trained YOLOv8 model weights
│
├── runs
│   └── detect                    # YOLO prediction outputs
│
├── src
│   ├── train_model.py            # Model training script
│   ├── detect_image.py           # Image detection script
│   ├── detect_video.py           # Video detection script
│   └── webcam_detection.py       # Real-time webcam detection
│
├── utils
│   └── helper.py                 # Helper functions (if needed)
│
├── main.py                       # Optional entry script
├── requirements.txt             # Python dependencies
└── README.md
```

---

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/Gopinath-chinnadurai/AI-Surveillance-System.git
cd AI-Surveillance-System
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit application

```bash
streamlit run app/streamlit_app.py
```

### 4. Open the application

After running, open the following in your browser:

```
http://localhost:8501
```

---

## Application Usage

The Streamlit interface provides the following options:

### Webcam Detection
- Opens the system webcam  
- Performs real-time surveillance detection  

### Image Upload
- Upload an image file  
- The system analyzes and predicts suspicious activity  

### Video Upload
- Upload a video file  
- Frames are processed sequentially for detection  

---

## Technologies Used

- Python  
- OpenCV  
- TensorFlow / Keras  
- NumPy  
- Streamlit  
- Git & GitHub  

---

## Future Improvements

With additional development time, the following improvements can be added:

- Real-time alert notifications  
- Email/SMS alerts  
- Cloud deployment  
- Multi-camera surveillance  
- Object tracking and behavior analysis  

---

## Requirements

- Python 3.10+
- Streamlit
- OpenCV
- NumPy
- TensorFlow / Keras

---

## Author

**Gopinath Chinnadurai**

GitHub:  
https://github.com/Gopinath-chinnadurai

LinkedIn:  
(Add your LinkedIn profile)

Portfolio:  
(Add your portfolio link)

---

## License

MIT License — for educational and demonstration purposes.
