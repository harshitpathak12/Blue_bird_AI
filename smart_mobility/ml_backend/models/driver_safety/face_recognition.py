"""
Face Recognition using ArcFace
"""
import cv2
import numpy as np
from typing import List, Optional, Tuple
import insightface
from ml_backend.config.model_configs import DRIVER_SAFETY_CONFIGS


class FaceRecognizer:
    """
    Face recognition using ArcFace pre-trained model
    """
    
    def __init__(self, config: Optional[dict] = None):
        """
        Initialize face recognizer
        
        Args:
            config: Configuration dictionary
        """
        if config is None:
            config = DRIVER_SAFETY_CONFIGS["face_recognition"]
        
        # Initialize InsightFace app
        self.app = insightface.app.FaceAnalysis(
            name=config.get("model_name", "arcface_r100_v1"),
            providers=['CPUExecutionProvider']  # Use GPU if available: ['CUDAExecutionProvider']
        )
        self.app.prepare(ctx_id=-1, det_size=(640, 640))
        
        self.embedding_size = config.get("embedding_size", 512)
        self.input_size = config.get("input_size", (112, 112))
        
        # Face database: {driver_id: embedding}
        self.face_database: dict = {}
    
    def get_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Get face embedding from face image
        
        Args:
            face_image: Face image (BGR format)
            
        Returns:
            Face embedding vector or None if no face detected
        """
        # Convert to RGB
        rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        
        # Get face embedding
        faces = self.app.get(rgb_image)
        
        if len(faces) == 0:
            return None
        
        # Return embedding of first (largest) face
        return faces[0].embedding
    
    def register_face(self, driver_id: str, face_image: np.ndarray) -> bool:
        """
        Register a face for a driver
        
        Args:
            driver_id: Unique driver identifier
            face_image: Face image to register
            
        Returns:
            True if successful, False otherwise
        """
        embedding = self.get_embedding(face_image)
        if embedding is None:
            return False
        
        self.face_database[driver_id] = embedding
        return True
    
    def verify(self, face_image: np.ndarray, driver_id: str, 
               threshold: float = 0.6) -> Tuple[bool, float]:
        """
        Verify if face matches registered driver
        
        Args:
            face_image: Face image to verify
            driver_id: Driver ID to verify against
            threshold: Similarity threshold
            
        Returns:
            (is_match, similarity_score)
        """
        if driver_id not in self.face_database:
            return False, 0.0
        
        embedding = self.get_embedding(face_image)
        if embedding is None:
            return False, 0.0
        
        # Calculate cosine similarity
        registered_embedding = self.face_database[driver_id]
        similarity = self._cosine_similarity(embedding, registered_embedding)
        
        is_match = similarity >= threshold
        return is_match, float(similarity)
    
    def identify(self, face_image: np.ndarray, 
                 threshold: float = 0.6) -> Optional[Tuple[str, float]]:
        """
        Identify face from database
        
        Args:
            face_image: Face image to identify
            threshold: Minimum similarity threshold
            
        Returns:
            (driver_id, similarity_score) or None if no match
        """
        embedding = self.get_embedding(face_image)
        if embedding is None:
            return None
        
        best_match = None
        best_score = 0.0
        
        for driver_id, registered_embedding in self.face_database.items():
            similarity = self._cosine_similarity(embedding, registered_embedding)
            if similarity > best_score and similarity >= threshold:
                best_score = similarity
                best_match = driver_id
        
        if best_match:
            return (best_match, float(best_score))
        return None
    
    @staticmethod
    def _cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings"""
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
