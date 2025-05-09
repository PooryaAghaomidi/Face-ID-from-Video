import numpy
import pandas
import logging
from typing import Optional
from deepface import DeepFace
from typing import List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class FaceIdentifier:
    def __init__(self, model_name: str, distance_metric: str, enforce_detection: bool, 
                 detector_backend: str, threshold: float) -> None:
        """
        Initializes the FaceIdentifier class with specified DeepFace parameters.

        :param model_name: Name of the DeepFace model to use.
        :param distance_metric: Metric to measure similarity.
        :param enforce_detection: Whether to enforce face detection.
        :param detector_backend: Backend used for face detection.
        :param threshold: Similarity threshold for face matching.
        """
        self.model_name = model_name
        self.distance_metric = distance_metric
        self.enforce_detection = enforce_detection
        self.detector_backend = detector_backend
        self.threshold = threshold

    def find_representation(self, images_list: List[str]) -> Optional[List[dict]]:
        """
        Extracts facial representations for a list of image paths.

        :param images_list: List of image paths.
        :return: List of representations or None if extraction fails.
        """
        try:
            representations = DeepFace.find_bulk_embeddings(
                employees=images_list,
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                enforce_detection=self.enforce_detection
            )
            return representations
        except Exception as e:
            logging.error(f"Error in find_representation: {e}")
            return None

    def identify_face(self, image: numpy.ndarray, documents: List[dict]) -> Optional[pandas.DataFrame]:
        """
        Identifies a face in the given image by comparing it against known representations.

        :param image: Input image as a numpy array.
        :param documents: List of precomputed face representations.
        :return: DataFrame of matching results or None if no match is found.
        """
        logging.info("Attempting to identify face in the frame.")

        try:
            results = DeepFace.find(
                img_path=image,
                representations=documents,
                model_name=self.model_name,
                distance_metric=self.distance_metric,
                enforce_detection=self.enforce_detection,
                detector_backend=self.detector_backend,
                threshold=self.threshold,
            )

            if not results or results[0].empty:
                logging.warning("No matching faces found.")
                return None

            logging.info("Face identified successfully.")
            return results[0]

        except Exception as e:
            logging.error(f"Error during face identification: {e}")
            return None
