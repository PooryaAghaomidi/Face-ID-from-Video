import torch
import numpy
import logging
from typing import Tuple, List, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger('ultralytics').setLevel(logging.CRITICAL)


class FaceDetector:
    def __init__(self, model: torch.nn.Module) -> None:
        """
        Initializes the FaceDetector class.

        :param model: A pre-trained face detection model.
        """
        self.model = model

    def detect_face(self, img: numpy.array) -> Union[Tuple[List[Tuple[int, int, int, int]], float], Tuple[None, None]]:
        """
        Detects the most confident face in an image and calculates the eye distance.

        :param img: Input image as a NumPy array.
        :return: A tuple containing the face bounding box coordinates and eye distance, or None if detection fails.
        """
        try:
            # Run the model on the image
            details = self.model(img)

            # Extract bounding boxes and keypoints
            faces = details[0].boxes.data  # [x1, y1, x2, y2, confidence]
            keypoints = details[0].keypoints.data  # Facial keypoints

            if faces.numel() == 0 or keypoints.numel() == 0:
                logging.warning("No faces detected in the image.")
                return None, None

            # Find the face with the highest confidence
            max_conf_idx = torch.argmax(faces[:, 4])

            # Extract face bounding box and scale it down by 0.25
            face_coords = faces[max_conf_idx]
            face_box = [(int(face_coords[0] * 0.25), int(face_coords[1] * 0.25),
                         int(face_coords[2] * 0.25), int(face_coords[3] * 0.25))]

            # Extract keypoints and compute eye distance
            keypoint_coords = keypoints[max_conf_idx]
            x1, y1 = keypoint_coords[0, :2]  # Left eye
            x2, y2 = keypoint_coords[1, :2]  # Right eye
            eye_distance = torch.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2).item()

            return face_box, eye_distance

        except Exception as e:
            logging.error(f"Error during face detection: {e}")
            return None, None
