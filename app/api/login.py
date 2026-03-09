"""
/api/login – Driver authentication (PDF: Face Detection + Recognition).
Login: face image → match → driver_id, driver_name, age.
Register: live photo only (base64 from camera) + name, age → store face_embedding.
"""

import base64
from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.schemas.payloads import LoginResponse, RegisterLiveBody, RegisterResponse
from app.services.driver_identity import match_embedding_to_driver
from database import driver_repository
from models.face_recongnition.face_recognition import ArcFaceModel

router = APIRouter(prefix="/api/login", tags=["login"])


# Instantiate ArcFace model once per process, but do not crash if the ONNX
# file is missing. This allows the API server to start even before the model
# file has been deployed.


try:
    _arcface_model: ArcFaceModel | None = ArcFaceModel()
except Exception as e:  # pragma: no cover - defensive startup guard
    print(f"[login] ArcFace model could not be initialized: {e}")
    _arcface_model = None


def _get_face_model() -> ArcFaceModel | None:
    return _arcface_model


def _extract_face_from_bytes(raw: bytes):
    """Decode image bytes and return largest face crop in BGR."""
    nparr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return None

    model = _get_face_model()
    if model is None:
        return None

    try:
        detections = model.detector.detect_faces(img)
    except Exception:
        return None
    if not detections:
        return None

    faces = list(detections.values())
    largest = max(
        faces,
        key=lambda d: (d["facial_area"][2] - d["facial_area"][0])
        * (d["facial_area"][3] - d["facial_area"][1]),
    )
    x1, y1, x2, y2 = largest["facial_area"]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
    face = img[y1:y2, x1:x2]
    if face.shape[0] < 50 or face.shape[1] < 50:
        return None
    return face


@router.post("/", response_model=LoginResponse)
async def login(
    driver_id: Optional[str] = Form(None),
    image: UploadFile = File(...),
):
    """
    Login with face image. If driver_id provided, verify against that driver; else match against DB.
    Returns driver_id, driver_name, age per PDF.
    """
    raw = await image.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty image")

    model = _get_face_model()
    if model is None:
        raise HTTPException(status_code=500, detail="Face model not available")

    face = _extract_face_from_bytes(raw)
    if face is None:
        raise HTTPException(status_code=400, detail="No valid face detected in image")

    emb = model.get_embedding(face)
    driver, score = match_embedding_to_driver(emb, driver_id=driver_id)
    if not driver:
        raise HTTPException(status_code=401, detail="Face not recognized or invalid driver_id")

    # Successful login
    driver_repository.update_last_seen(driver["driver_id"])
    return LoginResponse(
        driver_id=driver["driver_id"],
        driver_name=driver.get("name", ""),
        age=driver.get("age"),
        message="Login successful",
    )


@router.post("/register", response_model=RegisterResponse)
async def register(body: RegisterLiveBody):
    """
    First-time registration: live photo only (no file upload).
    Client must capture from camera and send base64-encoded image in JSON.
    Example: {"name": "John", "age": 30, "image_base64": "<base64 string>"}.
    """
    try:
        raw = base64.b64decode(body.image_base64, validate=True)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image_base64")

    if not raw or len(raw) == 0:
        raise HTTPException(status_code=400, detail="Empty image")

    model = _get_face_model()
    if model is None:
        raise HTTPException(status_code=500, detail="Face model not available")

    face = _extract_face_from_bytes(raw)
    if face is None:
        raise HTTPException(status_code=400, detail="No valid face detected. Use a live photo from the camera.")

    emb = model.get_embedding(face)
    face_embedding = emb.astype(float).tolist()

    driver = driver_repository.create_driver(
        driver_id=None,
        name=body.name,
        age=body.age,
        face_embedding=face_embedding,
        face_image_path=None,
    )
    return RegisterResponse(
        driver_id=driver["driver_id"],
        driver_name=body.name,
        age=body.age,
        message="Registration successful",
    )
