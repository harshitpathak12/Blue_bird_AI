"""
/api/login – Driver authentication (PDF: Face Detection + Recognition).
Login: face image → match (2D and/or 3D) → driver_id, driver_name, age.
Register: live photo (base64) + name, age → MediaPipe landmarks → 3D embedding → face_embedding_3d.
"""

import base64
from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.schemas.payloads import LoginResponse, RegisterLiveBody, RegisterResponse
from app.services.driver_identity import match_embedding_to_driver
from app.services.face_embedding_3d import build_3d_embedding
from database import driver_repository
from models.face_detection.face_detection import FaceDetector
from models.face_recongnition.face_recognition import ArcFaceModel

router = APIRouter(prefix="/api/login", tags=["login"])

try:
    _arcface_model: ArcFaceModel | None = ArcFaceModel()
except Exception as e:
    print(f"[login] ArcFace model could not be initialized: {e}")
    _arcface_model = None

_face_detector: FaceDetector | None = FaceDetector()


def _get_face_model() -> ArcFaceModel | None:
    return _arcface_model


def _decode_image(raw: bytes) -> Optional[np.ndarray]:
    """Decode image bytes to BGR frame."""
    nparr = np.frombuffer(raw, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


def _get_landmarks_from_image(img: np.ndarray):
    """Run MediaPipe Face Landmarker on full image; return (landmarks, img_w, img_h) or (None, None, None)."""
    if _face_detector is None:
        return None, None, None
    return _face_detector.get_landmarks(img)


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
    Uses 3D embedding (MediaPipe) when available and 2D (ArcFace) for legacy drivers.
    """
    raw = await image.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty image")

    img = _decode_image(raw)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    emb_3d = None
    landmarks, _, _ = _get_landmarks_from_image(img)
    if landmarks is not None:
        emb_3d = build_3d_embedding(landmarks)

    emb_2d = None
    model = _get_face_model()
    if model is not None:
        face = _extract_face_from_bytes(raw)
        if face is not None:
            emb_2d = model.get_embedding(face)

    if emb_3d is None and emb_2d is None:
        raise HTTPException(status_code=400, detail="No valid face detected in image")

    driver, score = match_embedding_to_driver(
        embedding_2d=emb_2d,
        embedding_3d=emb_3d,
        driver_id=driver_id,
    )
    if not driver:
        raise HTTPException(status_code=401, detail="Face not recognized or invalid driver_id")

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
    Uses MediaPipe Face Landmarker to get 3D landmarks and stores face_embedding_3d.
    """
    try:
        raw = base64.b64decode(body.image_base64, validate=True)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image_base64")

    if not raw or len(raw) == 0:
        raise HTTPException(status_code=400, detail="Empty image")

    img = _decode_image(raw)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    landmarks, _, _ = _get_landmarks_from_image(img)
    if landmarks is None:
        raise HTTPException(
            status_code=400,
            detail="No valid face detected. Use a live photo from the camera.",
        )

    emb_3d = build_3d_embedding(landmarks)
    if emb_3d is None:
        raise HTTPException(
            status_code=400,
            detail="Could not build face embedding. Ensure a clear face is visible.",
        )

    driver = driver_repository.create_driver(
        driver_id=None,
        name=body.name,
        age=body.age,
        face_embedding=None,
        face_embedding_3d=emb_3d.astype(float).tolist(),
        face_image_path=None,
    )
    return RegisterResponse(
        driver_id=driver["driver_id"],
        driver_name=body.name,
        age=body.age,
        message="Registration successful",
    )
