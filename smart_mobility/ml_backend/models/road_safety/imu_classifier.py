"""
IMU Classifier - Classify road events from IMU data using XGBoost/LSTM
"""
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Optional
import xgboost as xgb
from ml_backend.config.model_configs import ROAD_SAFETY_CONFIGS
from ml_backend.data.preprocessing import IMUPreprocessor


class IMUClassifier:
    """
    Classify road events (potholes, speed breakers, etc.) from IMU data
    Uses XGBoost for classification
    """
    
    def __init__(self, config: Optional[dict] = None):
        """
        Initialize IMU classifier
        
        Args:
            config: Configuration dictionary
        """
        if config is None:
            config = ROAD_SAFETY_CONFIGS["imu_classifier"]
        
        self.model = xgb.XGBClassifier(
            n_estimators=config.get("n_estimators", 100),
            max_depth=config.get("max_depth", 6),
            learning_rate=config.get("learning_rate", 0.1),
            random_state=42
        )
        
        self.preprocessor = IMUPreprocessor()
        
        # Load model if exists
        model_path = config.get("model_path")
        if model_path and Path(model_path).exists():
            self.model.load_model(str(model_path))
            self.is_trained = True
        else:
            self.is_trained = False
        
        # Event type mapping
        self.event_types = {
            0: "normal",
            1: "pothole",
            2: "speed_breaker",
            3: "rough_road",
            4: "smooth_road"
        }
    
    def extract_features(self, imu_sequence: np.ndarray) -> np.ndarray:
        """
        Extract features from IMU data sequence
        
        Args:
            imu_sequence: IMU data array [N, 6] (accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z)
            
        Returns:
            Feature vector
        """
        if len(imu_sequence) == 0:
            return np.zeros(24)  # Return zero features if empty
        
        # Preprocess
        normalized = self.preprocessor.normalize(imu_sequence)
        filtered = self.preprocessor.filter_noise(normalized)
        
        # Extract features
        features = []
        
        # Acceleration features
        accel = filtered[:, :3]
        accel_magnitude = np.linalg.norm(accel, axis=1)
        
        features.extend([
            np.mean(accel_magnitude),
            np.std(accel_magnitude),
            np.max(accel_magnitude),
            np.min(accel_magnitude),
            np.percentile(accel_magnitude, 25),
            np.percentile(accel_magnitude, 75)
        ])
        
        # Per-axis acceleration features
        for axis in range(3):
            features.extend([
                np.mean(accel[:, axis]),
                np.std(accel[:, axis]),
                np.max(np.abs(accel[:, axis]))
            ])
        
        # Gyroscope features
        gyro = filtered[:, 3:]
        gyro_magnitude = np.linalg.norm(gyro, axis=1)
        
        features.extend([
            np.mean(gyro_magnitude),
            np.std(gyro_magnitude),
            np.max(gyro_magnitude)
        ])
        
        return np.array(features)
    
    def predict(self, imu_sequence: np.ndarray) -> Dict:
        """
        Predict road event type from IMU sequence
        
        Args:
            imu_sequence: IMU data array [N, 6]
            
        Returns:
            Dictionary with prediction results
        """
        if not self.is_trained:
            return {
                "event_type": "unknown",
                "confidence": 0.0,
                "message": "Model not trained yet"
            }
        
        # Extract features
        features = self.extract_features(imu_sequence)
        features = features.reshape(1, -1)
        
        # Predict
        prediction = self.model.predict(features)[0]
        probabilities = self.model.predict_proba(features)[0]
        
        event_type = self.event_types.get(prediction, "unknown")
        confidence = float(np.max(probabilities))
        
        return {
            "event_type": event_type,
            "confidence": confidence,
            "probabilities": {
                self.event_types[i]: float(prob) 
                for i, prob in enumerate(probabilities)
            },
            "is_pothole": event_type == "pothole",
            "is_speed_breaker": event_type == "speed_breaker"
        }
    
    def predict_batch(self, imu_sequences: List[np.ndarray]) -> List[Dict]:
        """
        Predict for batch of IMU sequences
        
        Args:
            imu_sequences: List of IMU data arrays
            
        Returns:
            List of prediction results
        """
        results = []
        for sequence in imu_sequences:
            result = self.predict(sequence)
            results.append(result)
        return results
    
    def train(self, X: np.ndarray, y: np.ndarray):
        """
        Train the classifier
        
        Args:
            X: Feature array [N, features]
            y: Labels array [N]
        """
        # Extract features if raw IMU data provided
        if X.shape[1] == 6:  # Raw IMU data
            X_features = np.array([self.extract_features(x) for x in X])
        else:
            X_features = X
        
        # Train model
        self.model.fit(X_features, y)
        self.is_trained = True
    
    def save_model(self, model_path: str):
        """
        Save trained model
        
        Args:
            model_path: Path to save model
        """
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(model_path)
    
    def load_model(self, model_path: str):
        """
        Load trained model
        
        Args:
            model_path: Path to model file
        """
        if Path(model_path).exists():
            self.model.load_model(model_path)
            self.is_trained = True
