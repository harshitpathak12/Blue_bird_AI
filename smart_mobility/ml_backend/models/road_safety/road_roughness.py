"""
Road Roughness Index Calculation using Signal Processing
"""
import numpy as np
from scipy import signal
from typing import Dict, List, Optional
from ml_backend.config.model_configs import ROAD_SAFETY_CONFIGS


class RoadRoughnessCalculator:
    """
    Calculate road roughness index (IRI - International Roughness Index)
    from IMU acceleration data using signal processing
    """
    
    def __init__(self, config: Optional[dict] = None):
        """
        Initialize road roughness calculator
        
        Args:
            config: Configuration dictionary
        """
        if config is None:
            config = ROAD_SAFETY_CONFIGS["road_roughness"]
        
        self.window_size = config.get("window_size", 100)  # samples
        self.sampling_rate = config.get("sampling_rate", 100)  # Hz
        self.iri_thresholds = config.get("iri_thresholds", {
            "excellent": 1.0,
            "good": 2.0,
            "fair": 3.5,
            "poor": 5.0
        })
    
    def calculate_iri(self, vertical_acceleration: np.ndarray, 
                     speed: Optional[float] = None) -> float:
        """
        Calculate IRI (International Roughness Index) from vertical acceleration
        
        Args:
            vertical_acceleration: Vertical acceleration data (m/s²)
            speed: Vehicle speed (m/s), optional for more accurate calculation
            
        Returns:
            IRI value (m/km)
        """
        if len(vertical_acceleration) < self.window_size:
            return 0.0
        
        # Apply high-pass filter to remove low-frequency components
        nyquist = self.sampling_rate / 2
        high = 0.5 / nyquist  # 0.5 Hz cutoff
        b, a = signal.butter(4, high, btype='high')
        filtered_accel = signal.filtfilt(b, a, vertical_acceleration)
        
        # Calculate RMS (Root Mean Square) of filtered acceleration
        rms_accel = np.sqrt(np.mean(filtered_accel ** 2))
        
        # Convert to IRI (simplified formula)
        # More accurate calculation would use quarter-car simulation
        if speed is not None and speed > 0:
            # IRI approximation using speed
            iri = (rms_accel * 1000) / (speed ** 2)  # Simplified formula
        else:
            # Simplified IRI calculation without speed
            iri = rms_accel * 100  # Rough approximation
        
        return float(iri)
    
    def classify_roughness(self, iri: float) -> str:
        """
        Classify road roughness based on IRI value
        
        Args:
            iri: IRI value (m/km)
            
        Returns:
            Roughness category
        """
        if iri < self.iri_thresholds["excellent"]:
            return "excellent"
        elif iri < self.iri_thresholds["good"]:
            return "good"
        elif iri < self.iri_thresholds["fair"]:
            return "fair"
        elif iri < self.iri_thresholds["poor"]:
            return "poor"
        else:
            return "very_poor"
    
    def calculate_roughness_index(self, imu_data: np.ndarray, 
                                 speed: Optional[float] = None) -> Dict:
        """
        Calculate comprehensive roughness index from IMU data
        
        Args:
            imu_data: IMU data array [N, 6] (accel_x, accel_y, accel_z, ...)
            speed: Vehicle speed (m/s), optional
            
        Returns:
            Dictionary with roughness metrics
        """
        if len(imu_data) == 0:
            return {
                "iri": 0.0,
                "category": "unknown",
                "rms_accel": 0.0,
                "severity": "none"
            }
        
        # Extract vertical acceleration (Z-axis)
        vertical_accel = imu_data[:, 2]  # accel_z
        
        # Calculate IRI
        iri = self.calculate_iri(vertical_accel, speed)
        
        # Classify roughness
        category = self.classify_roughness(iri)
        
        # Calculate RMS acceleration
        rms_accel = float(np.sqrt(np.mean(vertical_accel ** 2)))
        
        # Determine severity
        if category in ["very_poor", "poor"]:
            severity = "high"
        elif category == "fair":
            severity = "medium"
        else:
            severity = "low"
        
        return {
            "iri": iri,
            "category": category,
            "rms_accel": rms_accel,
            "severity": severity,
            "is_rough": category in ["poor", "very_poor"]
        }
    
    def calculate_sliding_window(self, imu_data: np.ndarray, 
                                  window_size: Optional[int] = None,
                                  speed: Optional[float] = None) -> List[Dict]:
        """
        Calculate roughness index using sliding window
        
        Args:
            imu_data: IMU data array
            window_size: Window size for sliding calculation
            speed: Vehicle speed (m/s), optional
            
        Returns:
            List of roughness metrics for each window
        """
        if window_size is None:
            window_size = self.window_size
        
        results = []
        
        for i in range(0, len(imu_data) - window_size + 1, window_size // 2):
            window_data = imu_data[i:i + window_size]
            roughness = self.calculate_roughness_index(window_data, speed)
            roughness["window_start"] = i
            roughness["window_end"] = i + window_size
            results.append(roughness)
        
        return results
