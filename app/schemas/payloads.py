"""
API payload schemas per DRIVER SAFETY SYSTEM – COMPLETE PIPELINE DESIGN.
All request/response shapes align: MongoDB collections and API contracts.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# ---------- GPS (PDF: "GPS (Latitude + Longitude)" on alerts and geo-tagged events) ----------


class GPSLocation(BaseModel):
    latitude: float = Field(..., description="Latitude")
    longitude: float = Field(..., description="Longitude")


# ---------- /api/login (PDF: Face Recognition → driver_id, driver_name, age) ----------


class LoginRequest(BaseModel):
    """Payload for login: face image (base64 or multipart) + optional metadata."""

    driver_id: Optional[str] = Field(None, description="If provided, verify face against this driver")
    # Actual image is sent as multipart file or base64 in implementation


class LoginResponse(BaseModel):
    """Response after successful face recognition (PDF: driver_id, driver_name, age)."""

    driver_id: str
    driver_name: str
    age: Optional[int] = None
    message: str = "Login successful"


class RegisterRequest(BaseModel):
    """Payload for first-time registration (PDF: face image + name, age)."""

    name: str = Field(..., min_length=1)
    age: Optional[int] = Field(None, ge=1, le=120)
    # Face image sent as multipart or base64 in implementation


class RegisterLiveBody(BaseModel):
    """Registration with live-captured photo only (no file upload). image_base64 = base64-encoded JPEG/PNG from camera."""

    name: str = Field(..., min_length=1)
    age: Optional[int] = Field(None, ge=1, le=120)
    image_base64: str = Field(..., description="Base64-encoded image from live camera capture")


class RegisterResponse(BaseModel):
    """Response after driver registration (PDF: store face_embedding, face_image_path)."""

    driver_id: str
    driver_name: str
    age: Optional[int] = None
    message: str = "Registration successful"


# ---------- Session (PDF: session_id, driver_id, start_time, end_time) ----------


class SessionStartResponse(BaseModel):
    session_id: str
    driver_id: str
    start_time: datetime
    status: str = "active"
    message: str = "Session started"


class SessionEndResponse(BaseModel):
    session_id: str
    end_time: datetime
    status: str = "ended"
    message: str = "Session ended"


# ---------- Data Ingestion / Monitor (PDF: timestamp, driver_id; optional GPS) ----------


class FrameMetadata(BaseModel):
    """Metadata tagging per PDF: timestamp, driver_id; optional GPS."""

    timestamp: Optional[datetime] = None
    driver_id: str = Field(..., min_length=1)
    session_id: Optional[str] = None
    gps: Optional[GPSLocation] = None


class MonitorFrameRequest(BaseModel):
    """Request for /api/monitor: frame metadata. Frame bytes sent as multipart or WebSocket."""

    driver_id: str = Field(..., min_length=1)
    session_id: Optional[str] = None
    gps: Optional[GPSLocation] = None
    # timestamp can be server-generated if not sent


class MonitorFrameResponse(BaseModel):
    """Response after processing a frame: optional alert if fusion engine raised one."""

    processed: bool = True
    alert: Optional["AlertResponse"] = None
    driver_state: Optional[str] = None  # e.g. "normal", "fatigue", "distraction", "sleep"


# ---------- Alerts (PDF: driver_id, session_id, alert_type, confidence_score, timestamp, GPS) ----------


class AlertCreateBody(BaseModel):
    """Body to create/store an alert (PDF structure)."""

    driver_id: str = Field(..., min_length=1)
    session_id: Optional[str] = None
    alert_type: str = Field(..., description="e.g. fatigue, distraction, sleep")
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    gps: Optional[GPSLocation] = None


class AlertResponse(BaseModel):
    """Single alert as returned by API (PDF example: driver_id, alert_type, timestamp, GPS)."""

    alert_id: Optional[str] = None
    driver_id: str
    session_id: Optional[str] = None
    alert_type: str
    confidence_score: float
    timestamp: datetime
    gps: Optional[GPSLocation] = None


class AlertListResponse(BaseModel):
    """List of alerts for /api/alerts."""

    alerts: list[AlertResponse]
    total: int


# ---------- Safety Score (PDF: daily_scores – driver_id, date, fatigue_count, distraction_count, sleep_count, safety_score) ----------


class DailyScoreResponse(BaseModel):
    """One day's score per PDF: driver_id, date, fatigue_count, distraction_count, sleep_count, safety_score."""

    driver_id: str
    date: str = Field(..., description="Date in YYYY-MM-DD")
    fatigue_count: int = 0
    distraction_count: int = 0
    sleep_count: int = 0
    safety_score: float = Field(..., ge=0, le=100)
    risk_level: str = Field(..., description="Safe | Moderate Risk | High Risk")


class SafetyScoreResponse(BaseModel):
    """Response for /api/safety-score: one or more daily scores."""

    driver_id: str
    daily_scores: list[DailyScoreResponse]


class SafetyScoreQueryParams(BaseModel):
    """Query params for /api/safety-score (optional)."""

    driver_id: str = Field(..., min_length=1)
    date_from: Optional[str] = None  # YYYY-MM-DD
    date_to: Optional[str] = None   # YYYY-MM-DD


# Fix forward reference
MonitorFrameResponse.model_rebuild()
