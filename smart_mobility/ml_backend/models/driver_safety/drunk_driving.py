"""
Drunk Driving Pattern Detection using LSTM + Isolation Forest
"""
import numpy as np
from typing import List, Dict, Optional, Deque
from collections import deque
from pathlib import Path
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest
from ml_backend.config.model_configs import DRIVER_SAFETY_CONFIGS


class DrunkDrivingLSTM(nn.Module):
    """LSTM model for drunk driving pattern recognition"""
    
    def __init__(self, input_size: int = 6, hidden_size: int = 128, 
                 num_layers: int = 2, dropout: float = 0.2):
        super(DrunkDrivingLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out)


class DrunkDrivingDetector:
    """
    Drunk driving pattern detection using IMU data
    Combines Isolation Forest for anomaly detection and LSTM for pattern recognition
    """
    
    def __init__(self, config: Optional[dict] = None):
        """
        Initialize drunk driving detector
        
        Args:
            config: Configuration dictionary
        """
        if config is None:
            config = DRIVER_SAFETY_CONFIGS["drunk_driving"]
        
        self.sequence_length = config.get("sequence_length", 60)  # 60 seconds of data
        self.sampling_rate = 10  # 10 Hz (samples per second)
        self.buffer_size = self.sequence_length * self.sampling_rate
        
        # Initialize Isolation Forest
        self.isolation_forest = IsolationForest(
            contamination=config.get("contamination", 0.1),
            random_state=42
        )
        self.isolation_forest_trained = False
        
        # Initialize LSTM model
        self.lstm_units = config.get("lstm_units", 128)
        self.model = DrunkDrivingLSTM(input_size=6, hidden_size=self.lstm_units)
        
        # Load model if path exists
        model_path = config.get("model_path")
        if model_path and Path(model_path).exists():
            self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        
        self.model.eval()
        
        # Buffer for storing IMU data
        self.imu_buffer: deque = deque(maxlen=self.buffer_size)
        
        # Features to extract
        self.feature_names = [
            "accel_x", "accel_y", "accel_z",
            "gyro_x", "gyro_y", "gyro_z"
        ]
    
    def update(self, imu_data: Dict) -> Dict:
        """
        Update detector with new IMU data
        
        Args:
            imu_data: Dictionary with IMU readings
                Should contain: accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z
            
        Returns:
            Dictionary with detection results
        """
        # Extract features
        features = np.array([
            imu_data.get("accel_x", 0.0),
            imu_data.get("accel_y", 0.0),
            imu_data.get("accel_z", 0.0),
            imu_data.get("gyro_x", 0.0),
            imu_data.get("gyro_y", 0.0),
            imu_data.get("gyro_z", 0.0)
        ])
        
        self.imu_buffer.append(features)
        
        # Calculate features
        anomaly_score = 0.0
        pattern_score = 0.0
        
        if len(self.imu_buffer) >= self.buffer_size:
            # Calculate statistical features
            stats = self._calculate_statistics()
            
            # Anomaly detection
            if self.isolation_forest_trained:
                anomaly_score = self._detect_anomaly(stats)
            
            # Pattern recognition
            pattern_score = self._predict_pattern()
        
        # Combine scores
        combined_score = max(anomaly_score, pattern_score)
        is_drunk_driving = combined_score > 0.7
        
        return {
            "is_drunk_driving": is_drunk_driving,
            "anomaly_score": float(anomaly_score),
            "pattern_score": float(pattern_score),
            "combined_score": float(combined_score),
            "swerving_detected": self._detect_swerving(),
            "erratic_behavior": self._detect_erratic_behavior()
        }
    
    def _calculate_statistics(self) -> np.ndarray:
        """Calculate statistical features from IMU buffer"""
        if len(self.imu_buffer) == 0:
            return np.zeros(18)  # 6 features * 3 stats (mean, std, max)
        
        data = np.array(self.imu_buffer)
        
        # Calculate mean, std, max for each axis
        features = []
        for i in range(6):
            features.extend([
                np.mean(data[:, i]),
                np.std(data[:, i]),
                np.max(np.abs(data[:, i]))
            ])
        
        return np.array(features)
    
    def _detect_anomaly(self, stats: np.ndarray) -> float:
        """Detect anomaly using Isolation Forest"""
        if not self.isolation_forest_trained:
            return 0.0
        
        prediction = self.isolation_forest.predict([stats])
        score = self.isolation_forest.score_samples([stats])[0]
        
        # Convert to 0-1 scale (anomaly = 1, normal = 0)
        anomaly_score = 1.0 if prediction[0] == -1 else 0.0
        return anomaly_score
    
    def _predict_pattern(self) -> float:
        """Predict drunk driving pattern using LSTM"""
        if len(self.imu_buffer) < self.sequence_length * self.sampling_rate:
            return 0.0
        
        # Prepare sequence
        sequence = np.array(list(self.imu_buffer))
        # Reshape to (batch, seq_len, features)
        sequence = sequence[-self.sequence_length * self.sampling_rate:]
        sequence = sequence.reshape(1, -1, 6)
        
        # Convert to tensor
        input_tensor = torch.FloatTensor(sequence)
        
        # Predict
        with torch.no_grad():
            output = self.model(input_tensor)
            score = output.item()
        
        return score
    
    def _detect_swerving(self) -> bool:
        """Detect swerving behavior from gyroscope data"""
        if len(self.imu_buffer) < 10:
            return False
        
        data = np.array(list(self.imu_buffer)[-30:])  # Last 3 seconds
        gyro_y = data[:, 4]  # Y-axis gyroscope
        
        # Check for rapid left-right movements
        std_gyro = np.std(gyro_y)
        return std_gyro > 2.0  # Threshold for swerving
    
    def _detect_erratic_behavior(self) -> bool:
        """Detect erratic acceleration patterns"""
        if len(self.imu_buffer) < 10:
            return False
        
        data = np.array(list(self.imu_buffer)[-30:])
        accel = data[:, :3]  # Acceleration data
        
        # Check for sudden changes
        accel_changes = np.diff(accel, axis=0)
        max_change = np.max(np.abs(accel_changes))
        
        return max_change > 5.0  # Threshold for erratic behavior
    
    def train_isolation_forest(self, training_data: List[Dict]):
        """
        Train Isolation Forest on normal driving data
        
        Args:
            training_data: List of IMU data dictionaries
        """
        if len(training_data) < 100:
            return
        
        # Extract features
        features_list = []
        for imu_data in training_data:
            features = np.array([
                imu_data.get("accel_x", 0.0),
                imu_data.get("accel_y", 0.0),
                imu_data.get("accel_z", 0.0),
                imu_data.get("gyro_x", 0.0),
                imu_data.get("gyro_y", 0.0),
                imu_data.get("gyro_z", 0.0)
            ])
            features_list.append(features)
        
        # Calculate statistics for each sample
        stats_list = []
        for features in features_list:
            stats = np.array([
                np.mean(features),
                np.std(features),
                np.max(np.abs(features))
            ])
            stats_list.append(stats)
        
        # Train Isolation Forest
        X = np.array(stats_list)
        self.isolation_forest.fit(X)
        self.isolation_forest_trained = True
    
    def reset(self):
        """Reset detector state"""
        self.imu_buffer.clear()
