"""
Data Ingestion API - Receives data from mobile app/dashcam
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import numpy as np
import cv2
from datetime import datetime
import json
from ml_backend.config import settings
from ml_backend.data.storage.storage_manager import StorageManager

app = FastAPI(title="ML Backend Data Ingestion API", version="0.1.0")

# CORS middleware for mobile app access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize storage manager
storage_manager = StorageManager()


class IMUData(BaseModel):
    """IMU sensor data model"""
    accel_x: float = Field(..., description="Acceleration X-axis (m/s²)")
    accel_y: float = Field(..., description="Acceleration Y-axis (m/s²)")
    accel_z: float = Field(..., description="Acceleration Z-axis (m/s²)")
    gyro_x: float = Field(..., description="Gyroscope X-axis (rad/s)")
    gyro_y: float = Field(..., description="Gyroscope Y-axis (rad/s)")
    gyro_z: float = Field(..., description="Gyroscope Z-axis (rad/s)")
    timestamp: float = Field(..., description="Unix timestamp")
    device_id: str = Field(..., description="Device identifier")
    session_id: Optional[str] = Field(None, description="Driving session ID")


class GPSData(BaseModel):
    """GPS location data model"""
    latitude: float = Field(..., description="Latitude")
    longitude: float = Field(..., description="Longitude")
    altitude: Optional[float] = Field(None, description="Altitude (meters)")
    accuracy: Optional[float] = Field(None, description="GPS accuracy (meters)")
    speed: Optional[float] = Field(None, description="Speed (m/s)")
    timestamp: float = Field(..., description="Unix timestamp")
    device_id: str = Field(..., description="Device identifier")
    session_id: Optional[str] = Field(None, description="Driving session ID")


class EventData(BaseModel):
    """Event metadata"""
    event_type: str = Field(..., description="Type of event (harsh_braking, pothole, etc.)")
    severity: Optional[str] = Field(None, description="Event severity (low, medium, high, critical)")
    confidence: Optional[float] = Field(None, description="Detection confidence")
    timestamp: float = Field(..., description="Unix timestamp")
    device_id: str = Field(..., description="Device identifier")
    session_id: Optional[str] = Field(None, description="Driving session ID")
    gps_data: Optional[GPSData] = Field(None, description="GPS data at event time")
    imu_data: Optional[IMUData] = Field(None, description="IMU data at event time")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "0.1.0"
    }


@app.post("/ingest/imu")
async def ingest_imu(data: IMUData):
    """
    Receive IMU data from mobile app
    
    Args:
        data: IMU sensor readings
        
    Returns:
        Confirmation with stored data ID
    """
    try:
        # Store IMU data
        data_id = await storage_manager.store_imu_data(data.dict())
        
        return {
            "status": "success",
            "data_id": data_id,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error storing IMU data: {str(e)}")


@app.post("/ingest/gps")
async def ingest_gps(data: GPSData):
    """
    Receive GPS data from mobile app
    
    Args:
        data: GPS location data
        
    Returns:
        Confirmation with stored data ID
    """
    try:
        data_id = await storage_manager.store_gps_data(data.dict())
        
        return {
            "status": "success",
            "data_id": data_id,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error storing GPS data: {str(e)}")


@app.post("/ingest/image")
async def ingest_image(
    file: UploadFile = File(...),
    device_id: str = Field(..., description="Device identifier"),
    session_id: Optional[str] = None,
    timestamp: Optional[float] = None,
    event_type: Optional[str] = None
):
    """
    Receive image from mobile app/dashcam
    
    Args:
        file: Image file
        device_id: Device identifier
        session_id: Driving session ID
        timestamp: Event timestamp
        event_type: Type of event
        
    Returns:
        Confirmation with stored image ID
    """
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        # Store image
        metadata = {
            "device_id": device_id,
            "session_id": session_id,
            "timestamp": timestamp or datetime.now().timestamp(),
            "event_type": event_type,
            "filename": file.filename,
            "content_type": file.content_type
        }
        
        image_id = await storage_manager.store_image(image, metadata)
        
        return {
            "status": "success",
            "image_id": image_id,
            "shape": list(image.shape),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error storing image: {str(e)}")


@app.post("/ingest/video")
async def ingest_video(
    file: UploadFile = File(...),
    device_id: str = Field(..., description="Device identifier"),
    session_id: Optional[str] = None,
    timestamp: Optional[float] = None
):
    """
    Receive video clip from mobile app/dashcam
    
    Args:
        file: Video file
        device_id: Device identifier
        session_id: Driving session ID
        timestamp: Event timestamp
        
    Returns:
        Confirmation with stored video ID
    """
    try:
        contents = await file.read()
        
        metadata = {
            "device_id": device_id,
            "session_id": session_id,
            "timestamp": timestamp or datetime.now().timestamp(),
            "filename": file.filename,
            "content_type": file.content_type,
            "size_bytes": len(contents)
        }
        
        video_id = await storage_manager.store_video(contents, metadata)
        
        return {
            "status": "success",
            "video_id": video_id,
            "size_bytes": len(contents),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error storing video: {str(e)}")


@app.post("/ingest/event")
async def ingest_event(event: EventData):
    """
    Receive event data (combined IMU, GPS, and metadata)
    
    Args:
        event: Event data with all associated information
        
    Returns:
        Confirmation with stored event ID
    """
    try:
        event_id = await storage_manager.store_event(event.dict())
        
        return {
            "status": "success",
            "event_id": event_id,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error storing event: {str(e)}")


@app.post("/ingest/batch")
async def ingest_batch(data: List[dict]):
    """
    Receive batch of data (IMU, GPS, or events)
    
    Args:
        data: List of data items to store
        
    Returns:
        Confirmation with number of items stored
    """
    try:
        stored_count = await storage_manager.store_batch(data)
        
        return {
            "status": "success",
            "stored_count": stored_count,
            "total_count": len(data),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error storing batch: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.API_HOST, port=settings.API_PORT)
