"""
Eye Landmark Detection using MediaPipe
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional
import mediapipe as mp
from ml_backend.config.model_configs import DRIVER_SAFETY_CONFIGS


class EyeLandmarkDetector:
    """
    Eye landmark detection for fatigue analysis
    """
    
    def __init__(self, config: Optional[dict] = None):
        """
        Initialize eye landmark detector
        
        Args:
            config: Configuration dictionary
        """
        if config is None:
            config = DRIVER_SAFETY_CONFIGS["eye_landmark"]
        
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=config.get("min_detection_confidence", 0.5),
            min_tracking_confidence=config.get("min_tracking_confidence", 0.5)
        )
        
        # MediaPipe face mesh indices for left and right eyes
        # Left eye landmarks
        self.LEFT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        # Right eye landmarks
        self.RIGHT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    
    def detect_landmarks(self, image: np.ndarray) -> Optional[dict]:
        """
        Detect eye landmarks in image
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Dictionary with eye landmarks and EAR (Eye Aspect Ratio)
            or None if no face detected
        """
        # Convert to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process image
        results = self.face_mesh.process(rgb_image)
        
        if not results.multi_face_landmarks:
            return None
        
        # Get first face
        face_landmarks = results.multi_face_landmarks[0]
        
        # Extract eye landmarks
        h, w = image.shape[:2]
        left_eye = self._extract_eye_points(face_landmarks, self.LEFT_EYE_INDICES, w, h)
        right_eye = self._extract_eye_points(face_landmarks, self.RIGHT_EYE_INDICES, w, h)
        
        # Calculate EAR (Eye Aspect Ratio)
        left_ear = self._calculate_ear(left_eye)
        right_ear = self._calculate_ear(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0
        
        return {
            "left_eye": left_eye,
            "right_eye": right_eye,
            "left_ear": left_ear,
            "right_ear": right_ear,
            "avg_ear": avg_ear,
            "is_eye_open": avg_ear > 0.25  # Threshold for eye open/closed
        }
    
    def _extract_eye_points(self, landmarks, indices: List[int], 
                           width: int, height: int) -> List[Tuple[float, float]]:
        """Extract eye landmark points"""
        points = []
        for idx in indices:
            landmark = landmarks.landmark[idx]
            x = landmark.x * width
            y = landmark.y * height
            points.append((x, y))
        return points
    
    def _calculate_ear(self, eye_points: List[Tuple[float, float]]) -> float:
        """
        Calculate Eye Aspect Ratio (EAR)
        
        EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
        """
        if len(eye_points) < 6:
            return 0.0
        
        # Convert to numpy array
        points = np.array(eye_points)
        
        # Calculate distances
        # Vertical distances
        vertical_1 = np.linalg.norm(points[1] - points[5])
        vertical_2 = np.linalg.norm(points[2] - points[4])
        
        # Horizontal distance
        horizontal = np.linalg.norm(points[0] - points[3])
        
        # Calculate EAR
        if horizontal == 0:
            return 0.0
        
        ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
        return float(ear)
    
    def is_blinking(self, ear: float, threshold: float = 0.25) -> bool:
        """
        Determine if eye is blinking based on EAR
        
        Args:
            ear: Eye Aspect Ratio
            threshold: Threshold below which eye is considered closed
            
        Returns:
            True if blinking (eye closed), False otherwise
        """
        return ear < threshold
