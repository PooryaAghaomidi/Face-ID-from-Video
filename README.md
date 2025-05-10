# Face Identification System

## Introduction

The **Face Identification System** is a deep learning-based framework designed to detect and identify faces in videos.
This system uses **YOLO** for face detection, **DeepFace** for face recognition, and supports both **GPU and CPU**
execution. It allows you to:

- Extract frames from a video based on face detection.
- Compare detected faces with a reference image dataset.
- Retrieve matching images along with similarity scores.

---

## Installation

### **Step 1: Create a Python Environment**

This project requires **Python 3.9**. Create a new virtual environment:

```sh
python3.9 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### **Step 2: Install Dependencies**

Depending on whether you want to run the project on **GPU** or **CPU**, install the appropriate dependencies:

```text
⚠️ IMPORTANT NOTE ⚠️
To run the project on GPU, complete required steps based on your OS and GPU brand.
```

#### **GPU Installation** (CUDA 11.8)

```sh
pip install tensorflow-gpu==2.9
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118
```

#### **CPU Installation**

```sh
pip install tensorflow==2.9
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu
```

For OS-specific installation instructions, refer to:
- [PyTorch Previous Versions](https://pytorch.org/get-started/previous-versions/)
- [TensorFlow Installation](https://www.tensorflow.org/install/pip)

### **Step 3: Install Additional Requirements**

Once you've set up your environment, install the required dependencies:

```sh
pip install -r requirements.txt
```

---

## Usage

### Method 1: Use the code

#### **Step 1: Import the Face Identification Class**

```python
from face_sim import FaceIdentification
```

#### **Step 2: Initialize the Face Identification System**

Create an instance of `FaceIdentification` with your desired parameters:

```python
face_sim_class = FaceIdentification(
    face_detector_path="models/yolov8n-face.pt",  # Path to YOLO face detection model
    every_nth=5,  # Extract every 5th frame
    max_frames=1,  # Maximum frames to analyze per video
    model_name="Facenet512",  # Face recognition model
    distance_metric="euclidean_l2",  # Distance metric for similarity
    enforce_detection=True,  # Enforce face detection
    detector_backend="yolov8",  # Face detection backend
    threshold=None,  # Recognition threshold
    device="gpu",  # Device to run the model ("cpu" or "gpu")
    distance_threshold=0.9,  # Distance threshold for matches
    mongo_ip="localhost",  # MongoDB IP address
    mongo_port=27017  # MongoDB port
)
```

#### **Step 3: Manage Reference Images in Database**

Add images to the database:

```python
face_sim_class.update_db([
    "examples/References/_DSC0010.jpg",
    "examples/References/_DSC0619.jpg"
])
```

Remove specific identity or clear entire collection:

```python
# Remove a specific identity
face_sim_class.remove_identity(identity="specific_image")

# Clear all identities from database
face_sim_class.remove_identity()
```

#### **Step 4: Run Face Identification**

To identify faces in a video using the reference images stored in the database:

```python
result = face_sim_class.face_id(
    video_path="examples/Videos/video_2025-02-22_22-54-35.mp4"  # Path to input video
)
```

### Method 2: Use APIs

#### **Step 1: Start the API Service**

Run the API using either:

```bash
python api.py
```

OR (with Docker):

```bash
docker compose up --build
```

#### **Step 2: Access Methods**

- Gradio UI: Run app.py for a web interface

- Postman/curl: Send requests to http://localhost:5000

- Any HTTP client: Compatible with all standard API tools

## Notes

### **3. Important Note**

You can modify default configurations in `config.yaml` without changing code. 
The file supports all FaceIdentification parameters and API settings.

### **2. Example Output Format**

The face_id method returns a dictionary containing:

```python
{
    'time': {
        'video_processing_time': 3.74,  # Time taken for video processing
        'face_matching_time': 0.63  # Time taken for face matching
    },
    'image_path': [
        '_DSC0619.jpg',  # Matched reference images
        'DSC0494.jpg',
        'DSC0436.jpg'
    ]
}
```

### **3. API Endpoints**

#### `POST /initialize`

Initialize the FaceIdentification system with optional configuration.

**Request:**
```json
{
    "face_detector_path": "models/yolov8n-face.pt",
    "every_nth": 5,
    "max_frames": 1,
    "model_name": "Facenet512",
    ... (all FaceIdentification parameters)
}
```

#### `POST /update_db`

Add reference images via file upload or paths.

Methods:

1. File upload (multipart/form-data):

 - Field name: images

 - Multiple files supported

2. JSON payload:

**Request:**
```json
{"image_paths": ["path/to/image1.jpg", "path/to/image2.jpg"]}
```

#### `GET /get_identities`

List all stored identities.

**Request:**
```json
{"identities": ["person1", "person2"]}
```

#### `POST /remove_identity`

Remove specific identity or all identities.

**Request:**
```json
{"identity": "person_name"}  // Omit to remove all
```

#### `POST /face_id`

Process video for face identification.

Methods:

1. File upload (multipart/form-data):

 - Field name: video

2. JSON payload:

**Request:**
```json
{"video_path": "path/to/video.mp4"}
```
