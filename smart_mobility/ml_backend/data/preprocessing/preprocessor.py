"""
Data Preprocessing - Clean and prepare data for ML models
"""
import numpy as np
import cv2
from typing import Dict, Optional, Tuple, List
from scipy import signal
from scipy.ndimage import uniform_filter1d


class IMUPreprocessor:
    """Preprocess IMU sensor data"""
    
    def __init__(self, sampling_rate: int = 100):
        """
        Initialize IMU preprocessor
        
        Args:
            sampling_rate: Sampling rate in Hz
        """
        self.sampling_rate = sampling_rate
    
    def normalize(self, imu_data: np.ndarray) -> np.ndarray:
        """
        Normalize IMU data (zero mean, unit variance)
        
        Args:
            imu_data: IMU data array [N, 6] (accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z)
            
        Returns:
            Normalized IMU data
        """
        mean = np.mean(imu_data, axis=0)
        std = np.std(imu_data, axis=0)
        std = np.where(std == 0, 1, std)  # Avoid division by zero
        return (imu_data - mean) / std
    
    def filter_noise(self, imu_data: np.ndarray, 
                     filter_type: str = "butterworth") -> np.ndarray:
        """
        Filter noise from IMU data
        
        Args:
            imu_data: IMU data array
            filter_type: Type of filter ('butterworth', 'moving_average')
            
        Returns:
            Filtered IMU data
        """
        if filter_type == "butterworth":
            # Butterworth low-pass filter
            nyquist = self.sampling_rate / 2
            low = 20 / nyquist  # Cutoff at 20 Hz
            b, a = signal.butter(4, low, btype='low')
            
            filtered = np.zeros_like(imu_data)
            for i in range(imu_data.shape[1]):
                filtered[:, i] = signal.filtfilt(b, a, imu_data[:, i])
            
            return filtered
        
        elif filter_type == "moving_average":
            # Moving average filter
            window_size = 5
            return uniform_filter1d(imu_data, size=window_size, axis=0)
        
        return imu_data
    
    def detect_spikes(self, imu_data: np.ndarray, 
                     threshold: float = 3.0) -> np.ndarray:
        """
        Detect spikes in IMU data (potential potholes, speed breakers)
        
        Args:
            imu_data: IMU data array
            threshold: Standard deviation threshold for spike detection
            
        Returns:
            Boolean array indicating spikes
        """
        # Calculate magnitude of acceleration
        accel = imu_data[:, :3]
        accel_magnitude = np.linalg.norm(accel, axis=1)
        
        # Detect spikes using z-score
        mean = np.mean(accel_magnitude)
        std = np.std(accel_magnitude)
        
        spikes = np.abs(accel_magnitude - mean) > (threshold * std)
        
        return spikes
    
    def extract_features(self, imu_data: np.ndarray) -> Dict[str, float]:
        """
        Extract features from IMU data
        
        Args:
            imu_data: IMU data array
            
        Returns:
            Dictionary of extracted features
        """
        features = {}
        
        # Acceleration features
        accel = imu_data[:, :3]
        accel_magnitude = np.linalg.norm(accel, axis=1)
        
        features["accel_mean"] = float(np.mean(accel_magnitude))
        features["accel_std"] = float(np.std(accel_magnitude))
        features["accel_max"] = float(np.max(accel_magnitude))
        features["accel_min"] = float(np.min(accel_magnitude))
        features["accel_range"] = float(np.max(accel_magnitude) - np.min(accel_magnitude))
        
        # Gyroscope features
        gyro = imu_data[:, 3:]
        gyro_magnitude = np.linalg.norm(gyro, axis=1)
        
        features["gyro_mean"] = float(np.mean(gyro_magnitude))
        features["gyro_std"] = float(np.std(gyro_magnitude))
        features["gyro_max"] = float(np.max(gyro_magnitude))
        
        # Frequency domain features (FFT)
        fft_accel = np.fft.fft(accel_magnitude)
        fft_freq = np.fft.fftfreq(len(accel_magnitude), 1/self.sampling_rate)
        
        # Dominant frequency
        power = np.abs(fft_accel)
        dominant_freq_idx = np.argmax(power[1:]) + 1
        features["dominant_frequency"] = float(abs(fft_freq[dominant_freq_idx]))
        
        return features


class ImagePreprocessor:
    """Preprocess images for ML models"""
    
    def __init__(self, target_size: Optional[Tuple[int, int]] = None):
        """
        Initialize image preprocessor
        
        Args:
            target_size: Target size (width, height) for resizing
        """
        self.target_size = target_size
    
    def resize(self, image: np.ndarray, 
               size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Resize image
        
        Args:
            image: Input image
            size: Target size (width, height), uses self.target_size if None
            
        Returns:
            Resized image
        """
        if size is None:
            size = self.target_size
        
        if size is None:
            return image
        
        return cv2.resize(image, size)
    
    def normalize(self, image: np.ndarray, 
                  mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
                  std: Tuple[float, float, float] = (0.229, 0.224, 0.225)) -> np.ndarray:
        """
        Normalize image for deep learning models
        
        Args:
            image: Input image (BGR format)
            mean: Mean values for normalization
            std: Standard deviation values for normalization
            
        Returns:
            Normalized image
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Normalize
        image_norm = image_rgb.astype(np.float32) / 255.0
        image_norm = (image_norm - np.array(mean)) / np.array(std)
        
        return image_norm
    
    def augment(self, image: np.ndarray, 
               augmentations: List[str] = None) -> np.ndarray:
        """
        Apply data augmentations
        
        Args:
            image: Input image
            augmentations: List of augmentation types
            
        Returns:
            Augmented image
        """
        if augmentations is None:
            augmentations = []
        
        augmented = image.copy()
        
        if "flip" in augmentations:
            augmented = cv2.flip(augmented, 1)  # Horizontal flip
        
        if "brightness" in augmentations:
            # Random brightness adjustment
            alpha = np.random.uniform(0.8, 1.2)
            augmented = cv2.convertScaleAbs(augmented, alpha=alpha, beta=0)
        
        if "contrast" in augmentations:
            # Random contrast adjustment
            alpha = np.random.uniform(0.8, 1.2)
            augmented = cv2.convertScaleAbs(augmented, alpha=alpha, beta=0)
        
        return augmented
    
    def enhance(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance image quality
        
        Args:
            image: Input image
            
        Returns:
            Enhanced image
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge channels
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        return enhanced


class GPSPreprocessor:
    """Preprocess GPS data"""
    
    @staticmethod
    def validate_coordinates(latitude: float, longitude: float) -> bool:
        """
        Validate GPS coordinates
        
        Args:
            latitude: Latitude
            longitude: Longitude
            
        Returns:
            True if valid, False otherwise
        """
        return -90 <= latitude <= 90 and -180 <= longitude <= 180
    
    @staticmethod
    def calculate_distance(lat1: float, lon1: float, 
                          lat2: float, lon2: float) -> float:
        """
        Calculate distance between two GPS points (Haversine formula)
        
        Args:
            lat1, lon1: First point coordinates
            lat2, lon2: Second point coordinates
            
        Returns:
            Distance in meters
        """
        from math import radians, sin, cos, sqrt, atan2
        
        R = 6371000  # Earth radius in meters
        
        lat1_rad = radians(lat1)
        lat2_rad = radians(lat2)
        delta_lat = radians(lat2 - lat1)
        delta_lon = radians(lon2 - lon1)
        
        a = (sin(delta_lat / 2) ** 2 +
             cos(lat1_rad) * cos(lat2_rad) * sin(delta_lon / 2) ** 2)
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        
        return R * c
    
    @staticmethod
    def smooth_trajectory(coordinates: List[Tuple[float, float]], 
                         window_size: int = 5) -> List[Tuple[float, float]]:
        """
        Smooth GPS trajectory using moving average
        
        Args:
            coordinates: List of (latitude, longitude) tuples
            window_size: Window size for moving average
            
        Returns:
            Smoothed coordinates
        """
        if len(coordinates) < window_size:
            return coordinates
        
        smoothed = []
        for i in range(len(coordinates)):
            start = max(0, i - window_size // 2)
            end = min(len(coordinates), i + window_size // 2 + 1)
            
            window = coordinates[start:end]
            avg_lat = np.mean([c[0] for c in window])
            avg_lon = np.mean([c[1] for c in window])
            
            smoothed.append((avg_lat, avg_lon))
        
        return smoothed
