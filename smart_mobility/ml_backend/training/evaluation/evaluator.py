"""
Model Evaluation Framework
"""
import numpy as np
from typing import Dict, List, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from ml_backend.config.model_configs import SUCCESS_METRICS


class ModelEvaluator:
    """Evaluate models against success metrics"""
    
    def evaluate_driver_safety(self, y_true: np.ndarray, y_pred: np.ndarray,
                               y_proba: Optional[np.ndarray] = None) -> Dict:
        """Evaluate driver safety models"""
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
            "f1_score": float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
        }
        
        # Check against success metrics
        thresholds = SUCCESS_METRICS["driver_safety"]
        metrics["meets_fatigue_precision"] = metrics["precision"] >= thresholds["fatigue_precision_min"]
        metrics["meets_distraction_precision"] = metrics["precision"] >= thresholds["distraction_precision_min"]
        
        return metrics
    
    def evaluate_road_safety(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Evaluate road safety models"""
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
            "f1_score": float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
        }
        
        thresholds = SUCCESS_METRICS["road_safety"]
        metrics["meets_pothole_precision"] = metrics["precision"] >= thresholds["pothole_precision_min"]
        
        return metrics
