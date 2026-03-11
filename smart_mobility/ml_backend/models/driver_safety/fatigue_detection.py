"""
Fatigue Detection using PERCLOS + LSTM
"""
import numpy as np
from typing import List, Optional, Dict
from collections import deque
import torch
import torch.nn as nn
from ml_backend.config.model_configs import DRIVER_SAFETY_CONFIGS
from ml_backend.models.driver_safety.eye_landmark import EyeLandmarkDetector


class FatigueLSTM(nn.Module):
    """LSTM model for fatigue pattern recognition"""
    
    def __init__(self, input_size: int = 1, hidden_size: int = 64, 
                 num_layers: int = 2, dropout: float = 0.2):
        super(FatigueLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Take last output
        return self.sigmoid(out)


class FatigueDetector:
    """
    Fatigue detection using PERCLOS (Percentage of Eye Closure) and LSTM
    """
    
    def __init__(self, config: Optional[dict] = None):
        """
        Initialize fatigue detector
        
        Args:
            config: Configuration dictionary
        """
        if config is None:
            config = DRIVER_SAFETY_CONFIGS["fatigue_detection"]
        
        self.eye_detector = EyeLandmarkDetector()
        self.perclos_threshold = config.get("perclos_threshold", 0.5)
        self.sequence_length = config.get("sequence_length", 30)
        
        # Initialize LSTM model
        self.lstm_units = config.get("lstm_units", 64)
        self.model = FatigueLSTM(input_size=1, hidden_size=self.lstm_units)
        
        # Load model if path exists
        model_path = config.get("model_path")
        if model_path and model_path.exists():
            self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        
        self.model.eval()
        
        # Buffer for storing EAR values over time
        self.ear_buffer: deque = deque(maxlen=self.sequence_length)
        
        # PERCLOS calculation parameters
        self.eye_closed_threshold = 0.25
        self.window_size = 60  # frames (assuming 30 fps = 2 seconds)
    
    def update(self, image: np.ndarray) -> Dict:
        """
        Update fatigue detection with new frame
        
        Args:
            image: Current frame image
            
        Returns:
            Dictionary with fatigue metrics and status
        """
        # Detect eye landmarks
        eye_data = self.eye_detector.detect_landmarks(image)
        
        if eye_data is None:
            return {
                "fatigue_score": 0.0,
                "perclos": 0.0,
                "is_fatigued": False,
                "ear": 0.0
            }
        
        ear = eye_data["avg_ear"]
        self.ear_buffer.append(ear)
        
        # Calculate PERCLOS
        perclos = self._calculate_perclos()
        
        # Calculate fatigue score using LSTM if we have enough data
        fatigue_score = 0.0
        if len(self.ear_buffer) >= self.sequence_length:
            fatigue_score = self._predict_fatigue()
        
        # Determine fatigue status
        is_fatigued = (perclos > self.perclos_threshold) or (fatigue_score > 0.7)
        
        return {
            "fatigue_score": float(fatigue_score),
            "perclos": float(perclos),
            "is_fatigued": is_fatigued,
            "ear": float(ear),
            "eye_open": eye_data["is_eye_open"]
        }
    
    def _calculate_perclos(self) -> float:
        """
        Calculate PERCLOS (Percentage of Eye Closure)
        
        PERCLOS = (Time eyes closed) / (Total time) * 100
        """
        if len(self.ear_buffer) < 2:
            return 0.0
        
        # Count frames with eyes closed
        closed_frames = sum(1 for ear in self.ear_buffer 
                          if ear < self.eye_closed_threshold)
        
        total_frames = len(self.ear_buffer)
        perclos = closed_frames / total_frames if total_frames > 0 else 0.0
        
        return perclos
    
    def _predict_fatigue(self) -> float:
        """
        Predict fatigue using LSTM model
        
        Returns:
            Fatigue score between 0 and 1
        """
        if len(self.ear_buffer) < self.sequence_length:
            return 0.0
        
        # Prepare input sequence
        sequence = np.array(list(self.ear_buffer), dtype=np.float32)
        sequence = sequence.reshape(1, self.sequence_length, 1)  # (batch, seq, features)
        
        # Convert to tensor
        input_tensor = torch.FloatTensor(sequence)
        
        # Predict
        with torch.no_grad():
            output = self.model(input_tensor)
            fatigue_score = output.item()
        
        return fatigue_score
    
    def reset(self):
        """Reset detector state"""
        self.ear_buffer.clear()
