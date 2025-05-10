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

### **Step 1: Import the Face Identification Class**

```python
from face_sim import FaceIdentification
```

### **Step 2: Initialize the Face Identification System**

Create an instance of `FaceIdentification` with your desired parameters:

```python
face_sim_class = FaceIdentification(
    face_detector_path="path/to/yolo/model.pt",  # Path to YOLO model
    every_nth=1,  # Extract every nth frame
    max_frames=5,  # Maximum frames to analyze per video
    global_folder_path=None,  # Global reference folder (set to None if using face_id)
    model_name="Facenet512",  # Face recognition model
    distance_metric="euclidean_l2",  # Distance metric for similarity
    enforce_detection=True,  # Enforce face detection
    detector_backend="yolov8",  # Face detection backend
    threshold=None,  # Recognition threshold
    device="cpu",  # Device to run the model ("cpu" or "gpu")
    distance_threshold=0.9  # Distance threshold for matches
)
```

### **Step 3: Run Face Identification**

To identify faces in a video using a reference image folder:

```python
images_details = face_sim_class.face_id(
    video_path="path/to/your/video.mp4",  # Path to input video
    folder_path="path/to/folder/with/images/"  # Reference folder containing images
)
```

### **Example Output Format**

```json
[
    {"identity": "examples\\/References\\\\_DSC0436.jpg", "hash": "d86977889d2ea81639b9757ff234fdd10d9a3b28",
     "target_x": 1687, "target_y": 1335, "target_w": 671, "target_h": 975, "source_x": 181, "source_y": 247,
     "source_w": 153, "source_h": 237, "threshold": 1.04, "distance": 0.6998609872},
    
    {"identity": "examples\\/References\\\\_DSC0494.jpg", "hash": "a490a834601772798966003f6cf71362504e3d18",
     "target_x": 1452, "target_y": 2121, "target_w": 625, "target_h": 848, "source_x": 181, "source_y": 247,
     "source_w": 153, "source_h": 237, "threshold": 1.04, "distance": 0.7038239059},
    
    {"identity": "examples\\/References\\\\_DSC0619.jpg", "hash": "b79c6671cdd4a0ada1e5c309a480e5bd749a4967",
     "target_x": 1480, "target_y": 1945, "target_w": 788, "target_h": 1163, "source_x": 164, "source_y": 255,
     "source_w": 154, "source_h": 238, "threshold": 1.04, "distance": 0.7189472668},
    
    {"identity": "examples\\/References\\\\_DSC0628.jpg", "hash": "9d9eb772f6a5f92a80a375fb9e56bf0dfe0ed491",
     "target_x": 2097, "target_y": 2058, "target_w": 1164, "target_h": 1647, "source_x": 164, "source_y": 255,
     "source_w": 154, "source_h": 238, "threshold": 1.04, "distance": 0.860808437}
]
```

### **Important Notes:**

1. **Reference Folder Handling:**
    - If your reference folder is **fixed**, define it in `global_folder_path` when creating the instance.
    - If your reference folder **changes frequently**, set `folder_path` in `face_id`.
    - **One of these must be `None`** to avoid conflicts.

2. **First-Time Model Download:**
    - The first time you run the service, it may take a while to download the required models.
    - The first time you process a **new reference folder**, feature extraction might take time. However, if the folder
      remains unchanged, the system **will not reprocess the images** in future runs.
