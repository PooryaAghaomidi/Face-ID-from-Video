import cv2
import numpy as np
import logging
from typing import List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class VideoFrameExtractor:
    def __init__(self, face_detector, every_nth: int, max_frames: int) -> None:
        """
        Initializes the VideoFrameExtractor class.

        :param face_detector: An object with a detect_face method to detect faces in frames.
        :param every_nth: Extract every nth frame from the video.
        :param max_frames: Maximum number of frames to return based on face detection scores.
        """
        self.face_detector = face_detector
        self.every_nth = every_nth
        self.max_frames = max_frames

    def extract_frames(self, video_path: str) -> Optional[List[np.ndarray]]:
        """
        Extracts frames from a video where faces are detected and selects top frames based on distance.

        :param video_path: Path to the video file.
        :return: List of selected video frames as NumPy arrays, or None if extraction fails.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.error("Could not open video file: %s", video_path)
            return None

        frames = []
        distances = []
        face_coords_list = []
        frame_count = 0

        logging.info("Processing video: %s", video_path)

        while True:
            ret, frame = cap.read()
            if not ret:
                break  # End of video

            if frame_count % self.every_nth == 0:
                try:
                    detection_result = self.face_detector.detect_face(frame)
                    if detection_result:
                        face_coords, distance = detection_result
                        frames.append(frame)
                        distances.append(distance)
                        face_coords_list.append(face_coords)
                except Exception as e:
                    logging.warning(f"Error processing frame {frame_count}: {e}")

            frame_count += 1

        cap.release()

        if not distances:
            logging.warning("No faces detected in the video.")
            return None

        # Convert distances to a NumPy array
        distances = np.array(distances)

        # Select top frames based on distances (higher distance = higher ranking)
        top_indices = np.argsort(distances)[-self.max_frames:][::-1] if len(distances) >= self.max_frames else np.argsort(distances)[::-1]
        top_frames = [frames[idx] for idx in top_indices]

        logging.info("Successfully extracted %d frames.", len(top_frames))
        return top_frames
