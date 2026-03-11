"""
Storage Manager - Handles data storage for raw and processed data
"""
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List, Any
import numpy as np
import cv2
from ml_backend.config import settings
import asyncio


class StorageManager:
    """
    Manages storage for all data types:
    - IMU data (database)
    - GPS data (database)
    - Images (object storage)
    - Videos (object storage)
    - Events (database)
    """
    
    def __init__(self):
        self.raw_data_dir = settings.RAW_DATA_DIR
        self.processed_data_dir = settings.PROCESSED_DATA_DIR
        
        # Create directories
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Subdirectories
        self.images_dir = self.raw_data_dir / "images"
        self.videos_dir = self.raw_data_dir / "videos"
        self.imu_dir = self.raw_data_dir / "imu"
        self.gps_dir = self.raw_data_dir / "gps"
        self.events_dir = self.raw_data_dir / "events"
        
        # Create subdirectories
        for dir_path in [self.images_dir, self.videos_dir, self.imu_dir, 
                        self.gps_dir, self.events_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def _generate_id(self, data: Dict) -> str:
        """Generate unique ID for data"""
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.md5(data_str.encode()).hexdigest()
    
    async def store_imu_data(self, imu_data: Dict) -> str:
        """
        Store IMU data
        
        Args:
            imu_data: IMU sensor data dictionary
            
        Returns:
            Data ID
        """
        data_id = self._generate_id(imu_data)
        file_path = self.imu_dir / f"{data_id}.json"
        
        # Add metadata
        imu_data["_id"] = data_id
        imu_data["_stored_at"] = datetime.now().isoformat()
        
        # Store as JSON
        with open(file_path, 'w') as f:
            json.dump(imu_data, f, indent=2)
        
        return data_id
    
    async def store_gps_data(self, gps_data: Dict) -> str:
        """
        Store GPS data
        
        Args:
            gps_data: GPS location data dictionary
            
        Returns:
            Data ID
        """
        data_id = self._generate_id(gps_data)
        file_path = self.gps_dir / f"{data_id}.json"
        
        # Add metadata
        gps_data["_id"] = data_id
        gps_data["_stored_at"] = datetime.now().isoformat()
        
        # Store as JSON
        with open(file_path, 'w') as f:
            json.dump(gps_data, f, indent=2)
        
        return data_id
    
    async def store_image(self, image: np.ndarray, metadata: Dict) -> str:
        """
        Store image
        
        Args:
            image: Image as numpy array
            metadata: Image metadata
            
        Returns:
            Image ID
        """
        # Generate ID from metadata
        image_id = self._generate_id(metadata)
        
        # Save image
        image_path = self.images_dir / f"{image_id}.jpg"
        cv2.imwrite(str(image_path), image)
        
        # Save metadata
        metadata["_id"] = image_id
        metadata["_stored_at"] = datetime.now().isoformat()
        metadata["_image_path"] = str(image_path)
        
        metadata_path = self.images_dir / f"{image_id}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return image_id
    
    async def store_video(self, video_data: bytes, metadata: Dict) -> str:
        """
        Store video
        
        Args:
            video_data: Video file bytes
            metadata: Video metadata
            
        Returns:
            Video ID
        """
        video_id = self._generate_id(metadata)
        
        # Determine file extension from content type
        content_type = metadata.get("content_type", "video/mp4")
        ext = ".mp4" if "mp4" in content_type else ".avi"
        
        # Save video
        video_path = self.videos_dir / f"{video_id}{ext}"
        with open(video_path, 'wb') as f:
            f.write(video_data)
        
        # Save metadata
        metadata["_id"] = video_id
        metadata["_stored_at"] = datetime.now().isoformat()
        metadata["_video_path"] = str(video_path)
        
        metadata_path = self.videos_dir / f"{video_id}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return video_id
    
    async def store_event(self, event_data: Dict) -> str:
        """
        Store event data
        
        Args:
            event_data: Event data dictionary
            
        Returns:
            Event ID
        """
        event_id = self._generate_id(event_data)
        file_path = self.events_dir / f"{event_id}.json"
        
        # Add metadata
        event_data["_id"] = event_id
        event_data["_stored_at"] = datetime.now().isoformat()
        
        # Store as JSON
        with open(file_path, 'w') as f:
            json.dump(event_data, f, indent=2, default=str)
        
        return event_id
    
    async def store_batch(self, data_list: List[Dict]) -> int:
        """
        Store batch of data
        
        Args:
            data_list: List of data dictionaries
            
        Returns:
            Number of items stored
        """
        stored_count = 0
        
        for data in data_list:
            data_type = data.get("data_type", "event")
            
            try:
                if data_type == "imu":
                    await self.store_imu_data(data)
                elif data_type == "gps":
                    await self.store_gps_data(data)
                elif data_type == "event":
                    await self.store_event(data)
                else:
                    await self.store_event(data)  # Default to event
                
                stored_count += 1
            except Exception as e:
                print(f"Error storing data: {e}")
                continue
        
        return stored_count
    
    def get_image(self, image_id: str) -> Optional[np.ndarray]:
        """Retrieve image by ID"""
        image_path = self.images_dir / f"{image_id}.jpg"
        if image_path.exists():
            return cv2.imread(str(image_path))
        return None
    
    def get_imu_data(self, data_id: str) -> Optional[Dict]:
        """Retrieve IMU data by ID"""
        file_path = self.imu_dir / f"{data_id}.json"
        if file_path.exists():
            with open(file_path, 'r') as f:
                return json.load(f)
        return None
    
    def get_gps_data(self, data_id: str) -> Optional[Dict]:
        """Retrieve GPS data by ID"""
        file_path = self.gps_dir / f"{data_id}.json"
        if file_path.exists():
            with open(file_path, 'r') as f:
                return json.load(f)
        return None
    
    def get_event(self, event_id: str) -> Optional[Dict]:
        """Retrieve event by ID"""
        file_path = self.events_dir / f"{event_id}.json"
        if file_path.exists():
            with open(file_path, 'r') as f:
                return json.load(f)
        return None
