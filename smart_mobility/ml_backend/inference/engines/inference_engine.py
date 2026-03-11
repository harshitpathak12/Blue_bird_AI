"""
Inference Engine - Run models on new data
"""
from typing import Dict, Optional
import numpy as np
import cv2
from ml_backend.models.driver_safety import (
    FaceDetector, FatigueDetector, DistractionDetector
)
from ml_backend.models.road_safety import (
    IMUClassifier, PotholeDetector, CrackSegmenter
)


class InferenceEngine:
    """Unified inference engine for all models"""
    
    def __init__(self):
        self.face_detector = FaceDetector()
        self.fatigue_detector = FatigueDetector()
        self.distraction_detector = DistractionDetector()
        self.imu_classifier = IMUClassifier()
        self.pothole_detector = PotholeDetector()
        self.crack_segmenter = CrackSegmenter()
    
    def run_driver_safety(self, image: np.ndarray) -> Dict:
        """Run all driver safety models"""
        return {
            "face_detection": self.face_detector.detect(image),
            "fatigue": self.fatigue_detector.update(image),
            "distraction": self.distraction_detector.detect(image)
        }
    
    def run_road_safety(self, image: np.ndarray, imu_data: Optional[np.ndarray] = None) -> Dict:
        """Run all road safety models"""
        result = {
            "pothole": self.pothole_detector.detect(image),
            "crack": self.crack_segmenter.segment(image)
        }
        
        if imu_data is not None:
            result["imu_classifier"] = self.imu_classifier.predict(imu_data)
        
        return result
