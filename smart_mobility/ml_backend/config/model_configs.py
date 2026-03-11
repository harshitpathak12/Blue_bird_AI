"""
Model-specific configurations
"""
from typing import Dict, Any
from pathlib import Path
from ml_backend.config.settings import settings


# Driver Safety Model Configs
DRIVER_SAFETY_CONFIGS: Dict[str, Any] = {
    "face_detection": {
        "model_type": "mtcnn",
        "min_face_size": 40,
        "scale_factor": 0.709,
        "thresholds": [0.6, 0.7, 0.7],
        "model_path": None,  # Uses pre-trained
    },
    "face_recognition": {
        "model_type": "arcface",
        "model_name": "arcface_r100_v1",
        "input_size": (112, 112),
        "embedding_size": 512,
        "model_path": None,  # Downloads automatically
    },
    "eye_landmark": {
        "model_type": "mediapipe",
        "min_detection_confidence": 0.5,
        "min_tracking_confidence": 0.5,
    },
    "fatigue_detection": {
        "model_type": "perclos_lstm",
        "perclos_threshold": 0.5,
        "lstm_units": 64,
        "sequence_length": 30,  # frames
        "model_path": settings.MODELS_DIR / "fatigue_lstm.pth",
    },
    "distraction_detection": {
        "model_type": "yolov8",
        "model_size": "n",  # nano, small, medium, large
        "confidence_threshold": 0.5,
        "classes": ["cell phone", "person"],  # Classes to detect
        "model_path": settings.MODELS_DIR / "distraction_yolov8.pt",
    },
    "drunk_driving": {
        "model_type": "isolation_forest_lstm",
        "contamination": 0.1,
        "lstm_units": 128,
        "sequence_length": 60,  # seconds of IMU data
        "model_path": settings.MODELS_DIR / "drunk_driving.pth",
    },
}

# Road Safety Model Configs
ROAD_SAFETY_CONFIGS: Dict[str, Any] = {
    "imu_classifier": {
        "model_type": "xgboost",
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "model_path": settings.MODELS_DIR / "imu_classifier.pkl",
    },
    "pothole_vision": {
        "model_type": "yolov8",
        "model_size": "m",  # medium for better accuracy
        "confidence_threshold": 0.6,
        "iou_threshold": 0.5,
        "model_path": settings.MODELS_DIR / "pothole_yolov8.pt",
    },
    "crack_segmentation": {
        "model_type": "unet",
        "input_size": (512, 512),
        "num_classes": 2,  # background and crack
        "model_path": settings.MODELS_DIR / "crack_unet.pth",
    },
    "water_logging": {
        "model_type": "unet",
        "input_size": (512, 512),
        "num_classes": 2,
        "model_path": settings.MODELS_DIR / "water_logging_unet.pth",
    },
    "road_roughness": {
        "model_type": "signal_processing",
        "window_size": 100,  # samples
        "sampling_rate": 100,  # Hz
        "iri_thresholds": {
            "excellent": 1.0,
            "good": 2.0,
            "fair": 3.5,
            "poor": 5.0,
        },
    },
}

# Event Fusion Config
FUSION_CONFIG: Dict[str, Any] = {
    "temporal_window": 5.0,  # seconds
    "spatial_threshold": 50.0,  # meters
    "confidence_weights": {
        "driver_safety": 0.6,
        "road_safety": 0.4,
    },
    "severity_levels": {
        "low": 0.0,
        "medium": 0.3,
        "high": 0.6,
        "critical": 0.8,
    },
    "h3_resolution": 9,  # For geo-clustering
}

# Success Metrics Thresholds (from Success Metrics document)
SUCCESS_METRICS: Dict[str, Dict[str, float]] = {
    "driver_safety": {
        "fatigue_precision_min": 0.85,
        "distraction_precision_min": 0.85,
        "false_alert_rate_max": 0.15,
        "missed_critical_max": 0.10,
    },
    "road_safety": {
        "pothole_precision_min": 0.85,
        "false_pothole_rate_max": 0.20,
        "severity_accuracy_min": 0.80,
        "deduplication_rate_min": 0.70,
    },
    "platform": {
        "uptime_min": 0.995,
        "gps_validity_min": 0.95,
        "sensor_integrity_min": 0.95,
        "media_usability_min": 0.80,
    },
}
