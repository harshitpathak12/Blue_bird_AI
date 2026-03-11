"""
Pothole Detection using YOLOv8
"""
import cv2
import numpy as np
from typing import List, Dict, Optional
from ultralytics import YOLO
from pathlib import Path
from ml_backend.config.model_configs import ROAD_SAFETY_CONFIGS


class PotholeDetector:
    """
    Detect potholes in road images using YOLOv8
    Can use pre-trained model or fine-tuned model
    """
    
    def __init__(self, config: Optional[dict] = None):
        """
        Initialize pothole detector
        
        Args:
            config: Configuration dictionary
        """
        if config is None:
            config = ROAD_SAFETY_CONFIGS["pothole_vision"]
        
        self.confidence_threshold = config.get("confidence_threshold", 0.6)
        self.iou_threshold = config.get("iou_threshold", 0.5)
        model_size = config.get("model_size", "m")  # medium
        model_path = config.get("model_path")
        
        # Initialize YOLOv8 model
        if model_path and Path(model_path).exists():
            # Load custom trained model
            self.model = YOLO(str(model_path))
        else:
            # Load pre-trained YOLOv8 model
            # Note: For production, you should fine-tune on pothole dataset
            self.model = YOLO(f"yolov8{model_size}.pt")
    
    def detect(self, image: np.ndarray) -> Dict:
        """
        Detect potholes in image
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Dictionary with detection results
        """
        # Run inference
        results = self.model(
            image, 
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            verbose=False
        )
        
        detections = []
        
        if len(results) > 0:
            result = results[0]
            
            for box in result.boxes:
                confidence = float(box.conf[0])
                
                if confidence >= self.confidence_threshold:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # Get class name
                    class_id = int(box.cls[0])
                    class_name = result.names[class_id]
                    
                    detections.append({
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "confidence": confidence,
                        "class": class_name,
                        "class_id": class_id,
                        "area": float((x2 - x1) * (y2 - y1))
                    })
        
        # Calculate severity based on number and size of potholes
        severity = self._calculate_severity(detections, image.shape)
        
        return {
            "potholes_detected": len(detections),
            "detections": detections,
            "severity": severity,
            "has_potholes": len(detections) > 0
        }
    
    def _calculate_severity(self, detections: List[Dict], 
                           image_shape: tuple) -> str:
        """
        Calculate severity based on pothole detections
        
        Args:
            detections: List of detection dictionaries
            image_shape: Image shape (height, width, channels)
            
        Returns:
            Severity level (low, medium, high, critical)
        """
        if len(detections) == 0:
            return "none"
        
        image_area = image_shape[0] * image_shape[1]
        total_pothole_area = sum(d["area"] for d in detections)
        coverage_ratio = total_pothole_area / image_area
        
        # Severity based on coverage and count
        if len(detections) >= 5 or coverage_ratio > 0.1:
            return "critical"
        elif len(detections) >= 3 or coverage_ratio > 0.05:
            return "high"
        elif len(detections) >= 2 or coverage_ratio > 0.02:
            return "medium"
        else:
            return "low"
    
    def detect_batch(self, images: List[np.ndarray]) -> List[Dict]:
        """
        Detect potholes in batch of images
        
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
    
    def visualize(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Visualize pothole detections on image
        
        Args:
            image: Original image
            detections: List of detection dictionaries
            
        Returns:
            Image with bounding boxes drawn
        """
        vis_image = image.copy()
        
        for det in detections:
            x1, y1, x2, y2 = map(int, det["bbox"])
            confidence = det["confidence"]
            
            # Draw bounding box
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # Draw label
            label = f"Pothole {confidence:.2f}"
            cv2.putText(vis_image, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        return vis_image
