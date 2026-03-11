"""
Distraction Detection using YOLOv8
"""
import cv2
import numpy as np
from typing import List, Dict, Optional
from ultralytics import YOLO
from pathlib import Path
from ml_backend.config.model_configs import DRIVER_SAFETY_CONFIGS


class DistractionDetector:
    """
    Distraction detection using YOLOv8
    Detects phone usage, eating, and other distractions
    """
    
    def __init__(self, config: Optional[dict] = None):
        """
        Initialize distraction detector
        
        Args:
            config: Configuration dictionary
        """
        if config is None:
            config = DRIVER_SAFETY_CONFIGS["distraction_detection"]
        
        self.confidence_threshold = config.get("confidence_threshold", 0.5)
        self.target_classes = config.get("classes", ["cell phone", "person"])
        
        # Initialize YOLOv8 model
        model_size = config.get("model_size", "n")  # nano, small, medium, large
        model_path = config.get("model_path")
        
        if model_path and Path(model_path).exists():
            # Load custom trained model
            self.model = YOLO(str(model_path))
        else:
            # Load pre-trained YOLOv8 model
            self.model = YOLO(f"yolov8{model_size}.pt")
        
        # Map class names to IDs
        self.class_names = self.model.names
        self.target_class_ids = [
            i for i, name in self.class_names.items() 
            if name.lower() in [c.lower() for c in self.target_classes]
        ]
    
    def detect(self, image: np.ndarray) -> Dict:
        """
        Detect distractions in image
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Dictionary with detection results
        """
        # Run inference
        results = self.model(image, conf=self.confidence_threshold, verbose=False)
        
        detections = []
        is_distracted = False
        distraction_type = None
        
        if len(results) > 0:
            result = results[0]
            
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                
                # Check if it's a target class
                if class_id in self.target_class_ids:
                    class_name = self.class_names[class_id]
                    
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    detections.append({
                        "class": class_name,
                        "class_id": class_id,
                        "confidence": confidence,
                        "bbox": [float(x1), float(y1), float(x2), float(y2)]
                    })
                    
                    # Phone detection is a strong indicator of distraction
                    if "phone" in class_name.lower() and confidence > 0.6:
                        is_distracted = True
                        distraction_type = "phone_usage"
        
        # Determine distraction status
        if not is_distracted and len(detections) > 0:
            # Check for other distraction patterns
            phone_detections = [d for d in detections if "phone" in d["class"].lower()]
            if phone_detections:
                is_distracted = True
                distraction_type = "phone_usage"
        
        return {
            "is_distracted": is_distracted,
            "distraction_type": distraction_type,
            "detections": detections,
            "confidence": max([d["confidence"] for d in detections], default=0.0)
        }
    
    def detect_batch(self, images: List[np.ndarray]) -> List[Dict]:
        """
        Detect distractions in batch of images
        
        Args:
            images: List of input images
            
        Returns:
            List of detection results
        """
        results = []
        for image in images:
            detection = self.detect(image)
            results.append(detection)
        return results
