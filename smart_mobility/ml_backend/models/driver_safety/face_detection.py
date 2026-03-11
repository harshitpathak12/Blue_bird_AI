"""
Face Detection using MTCNN
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional
from mtcnn import MTCNN
from ml_backend.config.model_configs import DRIVER_SAFETY_CONFIGS


class FaceDetector:
    """
    Face detection using MTCNN pre-trained model
    """
    
    def __init__(self, config: Optional[dict] = None):
        """
        Initialize face detector
        
        Args:
            config: Configuration dictionary, uses default if None
        """
        if config is None:
            config = DRIVER_SAFETY_CONFIGS["face_detection"]
        
        # MTCNN parameters - some versions don't accept thresholds directly
        mtcnn_params = {
            "min_face_size": config.get("min_face_size", 40),
            "scale_factor": config.get("scale_factor", 0.709)
        }
        # Only add thresholds if the version supports it
        thresholds = config.get("thresholds", [0.6, 0.7, 0.7])
        try:
            self.detector = MTCNN(**mtcnn_params, thresholds=thresholds)
        except TypeError:
            # Older MTCNN version doesn't support thresholds parameter
            self.detector = MTCNN(**mtcnn_params)
    
    def detect(self, image: np.ndarray) -> List[dict]:
        """
        Detect faces in image
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            List of face detections with bounding boxes and landmarks
            Each detection contains:
            - 'box': [x, y, width, height]
            - 'confidence': detection confidence
            - 'keypoints': facial landmarks
        """
        # MTCNN expects RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image
        
        detections = self.detector.detect_faces(rgb_image)
        return detections
    
    def detect_batch(self, images: List[np.ndarray]) -> List[List[dict]]:
        """
        Detect faces in batch of images
        
        Args:
            images: List of input images
            
        Returns:
            List of detection results for each image
        """
        results = []
        for image in images:
            detections = self.detect(image)
            results.append(detections)
        return results
    
    def extract_face(self, image: np.ndarray, detection: dict, 
                     margin: int = 10) -> Optional[np.ndarray]:
        """
        Extract face region from image
        
        Args:
            image: Original image
            detection: Face detection result
            margin: Margin to add around face
            
        Returns:
            Extracted face image or None if invalid
        """
        x, y, w, h = detection['box']
        
        # Add margin
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(image.shape[1] - x, w + 2 * margin)
        h = min(image.shape[0] - y, h + 2 * margin)
        
        face = image[y:y+h, x:x+w]
        return face if face.size > 0 else None
