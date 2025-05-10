import os
import sys

# Make sure 'modules' is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'modules')))

import yaml
import json
import torch
import logging
import pandas as pd
from time import time
import tensorflow as tf
from typing import Optional
from ultralytics import YOLO
from pymongo import MongoClient
from typing import Optional, List
from source.detection import FaceDetector
from source.preprocess import VideoFrameExtractor
from source.identification import FaceIdentifier

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load config from YAML
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


class FaceIdentification:
    def __init__(
        self,
        face_detector_path: str = config["face_detector_path"],
        mongo_ip: str = config["mongo_ip"],
        mongo_port: int = config["mongo_port"],
        every_nth: int = config["every_nth"],
        max_frames: int = config["max_frames"],
        model_name: str = config["model_name"],
        distance_metric: str = config["distance_metric"],
        enforce_detection: bool = config["enforce_detection"],
        detector_backend: str = config["detector_backend"],
        threshold: Optional[float] = config["threshold"],
        device: str = config["device"],
        distance_threshold: float = config["distance_threshold"],
    ) -> None:
        """
        Initializes the FaceIdentification system with all necessary components:
        detector, identifier, database, and preprocessing pipeline.

        :param face_detector_path: Path to the YOLO face detector model.
        :param mongo_ip: MongoDB host IP.
        :param mongo_port: MongoDB port.
        :param every_nth: Extract every nth frame from the video.
        :param max_frames: Maximum frames to extract and analyze.
        :param model_name: DeepFace recognition model.
        :param distance_metric: Similarity metric.
        :param enforce_detection: Enforce that a face must be detected.
        :param detector_backend: Backend used for face detection.
        :param threshold: Confidence threshold (DeepFace).
        :param device: Device for inference ('gpu' or 'cpu').
        :param distance_threshold: Max distance to consider a valid match.
        """
        # Validate parameters
        self._validate_parameters(model_name, distance_metric, detector_backend, device)
        self.distance_threshold = distance_threshold

        # Initialize face detection, preprocessing, and identification modules
        try:
            model = YOLO(model=face_detector_path)
            self.face_detector = FaceDetector(model=model)
        except Exception as e:
            logging.error(f"Failed to initialize face detector: {e}")
            raise

        self.preprocessor = VideoFrameExtractor(face_detector=self.face_detector,
                                                every_nth=every_nth,
                                                max_frames=max_frames)
        self.identification = FaceIdentifier(model_name=model_name,
                                             distance_metric=distance_metric,
                                             enforce_detection=enforce_detection,
                                             detector_backend=detector_backend,
                                             threshold=threshold)
        
        # Connect to the database
        try:
            client = MongoClient(f"mongodb://{mongo_ip}:{mongo_port}/")
            db = client["gallery"]
            self.collection = db["images"]
            self.documents = list(self.collection.find())
        except Exception as e:
            logging.error(f"Failed to connect to MongoDB: {e}")
            raise

    def _validate_parameters(self, model_name: str, distance_metric: str, detector_backend: str, device: str) -> None:
        """
        Validates model, metric, backend, and device parameters.

        :raises ValueError: If an invalid parameter is provided.
        """
        models = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib", "SFace", "GhostFaceNet"]
        metrics = ["cosine", "euclidean", "euclidean_l2"]
        backends = ['opencv', 'ssd', 'dlib', 'mtcnn', 'fastmtcnn', 'retinaface', 'mediapipe', 'yolov8', 'yolov11s', 'yolov11n', 'yolov11m', 'yunet', 'centerface']

        if model_name not in models:
            raise ValueError(f"Invalid model. Choose from: {models}")
        if distance_metric not in metrics:
            raise ValueError(f"Invalid metric. Choose from: {metrics}")
        if detector_backend not in backends:
            raise ValueError(f"Invalid face detection backend. Choose from: {backends}")
        if device == "gpu" and not tf.test.is_gpu_available() and not torch.cuda.is_available():
            raise ValueError("GPU is not available. Check CUDA, cuDNN, and GPU drivers.")
        
    def update_db(self, images_list: List[str]) -> None:
        """
        Updates the face database with new image representations.

        :param images_list: List of image paths to extract representations from.
        """
        try:
            representations = self.identification.find_representation(images_list=images_list)
            if representations:
                self.collection.insert_many(representations)
                self.documents = list(self.collection.find())
                logging.info(f"{len(representations)} representations added to database.")
            else:
                logging.warning("No representations found for given images.")
        except Exception as e:
            logging.error(f"Error while updating database: {e}")

    def get_identities(self) -> List[str]:
        """
        Retrieves a list of all 'identity' values from the database.

        :return: List of identity strings.
        """
        return [os.path.basename(doc['identity']) for doc in self.collection.find({}, {'identity': 1, '_id': 0})]
    
    def remove_identity(self, identity: Optional[str] = None) -> None:
        """
        Removes a specific identity or clears the entire collection.

        :param identity: Identity string to remove. If None, deletes all documents.
        """
        if identity:
            result = self.collection.delete_many({'identity': identity})
            logging.info(f"Removed {result.deleted_count} documents with identity: {identity}")
        else:
            result = self.collection.delete_many({})
            logging.info(f"Removed all documents from the collection.")
        self.documents = list(self.collection.find())

    def face_id(self, video_path: str) -> dict:
        """
        Identifies faces from a video by matching them against a database.

        :param video_path: Path to the input video.
        :return: Dictionary containing identified faces and processing times.
                Format: {
                    "time": {
                        "video_processing_time": float,
                        "face_matching_time": float
                    },
                    "image_path": list[str]  # list of matched identities
                }
        """
        logging.info(f"Extracting frames from video: {video_path}")
        response = {
            "time": {},
            "image_path": []
        }

        start_video_processing = time()
        try:
            top_frames = self.preprocessor.extract_frames(video_path)
        except Exception as e:
            logging.error(f"Error during frame extraction: {e}")
            return response
        response["time"]["video_processing_time"] = time() - start_video_processing

        if not top_frames:
            logging.warning("No frames extracted, returning empty result.")
            return response

        start_face_matching = time()
        matched_faces = []
        for frame in top_frames:
            try:
                df = self.identification.identify_face(image=frame, documents=self.documents)
                if df is not None:
                    matched_faces.append(df)
            except Exception as e:
                logging.error(f"Error identifying face in frame: {e}")
        response["time"]["face_matching_time"] = time() - start_face_matching

        if not matched_faces:
            logging.info("No faces matched, returning empty result.")
            return response

        try:
            final_df = pd.concat(matched_faces, ignore_index=True)
            filtered_df = final_df[final_df["distance"] < self.distance_threshold]
            filtered_df = filtered_df.sort_values(by="distance").drop_duplicates(subset="identity", keep="first")

            logging.info("Face identification completed successfully.")
            response["image_path"] = [os.path.basename(identity) for identity in filtered_df['identity']]
        except Exception as e:
            logging.error(f"Error processing final DataFrame: {e}")
            
        return response
        