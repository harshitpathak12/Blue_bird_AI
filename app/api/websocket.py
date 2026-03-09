import asyncio
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.fusion import FusionEngine, ModelOutputs
from app.utils.overlay import draw_driver_hud
from app.services.driver_identity import match_embedding_to_driver
from database.session_repository import create_session, end_session
from database.alert_repository import insert_alert
from models.face_detection.face_detection import FaceDetector
from models.fatigue_detection.fatigue_detection_model import ModelFatigue
from models.distraction_detection.model import ModelHeadPose
from models.blink_perclos.model import ModelEyeGaze
from models.blink_perclos.drowsiness_model import ModelDrowsiness

router = APIRouter()

# Fixed low resolution for entire pipeline to minimize latency (landmarks + models + encode)
STREAM_WIDTH, STREAM_HEIGHT = 320, 240
STREAM_JPEG_QUALITY = 35  # Lower = faster encode, smaller payload
# Run full model pipeline every Nth frame; 1 = every frame, 2 = half the frames (reuse last result)
MODEL_INTERVAL = 2

# Thread pool: decode, landmarks, 4 models, draw+encode, ArcFace, DB
_executor = ThreadPoolExecutor(max_workers=10, thread_name_prefix="stream_worker")

# Instantiate models and fusion engine once (per process)
face_detector = FaceDetector()
fatigue_model = ModelFatigue()
headpose_model = ModelHeadPose()
eye_gaze_model = ModelEyeGaze()
drowsiness_model = ModelDrowsiness()
fusion_engine = FusionEngine()

# Face recognition (ArcFace): optional; pipeline works without it
try:
    from models.face_recongnition.face_recognition import ArcFaceModel
    _arcface_model: ArcFaceModel | None = ArcFaceModel()
except Exception as e:
    print(f"[stream] ArcFace not loaded: {e}")
    _arcface_model = None


def _run_fatigue(frame_copy, landmarks, img_w, img_h):
    return fatigue_model.process(frame_copy, landmarks, img_w, img_h)


def _run_drowsiness(frame_copy, landmarks, img_w, img_h):
    return drowsiness_model.process(frame_copy, landmarks, img_w, img_h)


def _run_headpose(frame_copy, landmarks, img_w, img_h):
    return headpose_model.process(frame_copy, landmarks, img_w, img_h)


def _run_eye_gaze(frame_copy, landmarks, img_w, img_h):
    return eye_gaze_model.process(frame_copy, landmarks, img_w, img_h)


def _get_landmarks_sync(frame):
    """Run in thread so event loop isn't blocked by MediaPipe."""
    return face_detector.get_landmarks(frame)


def _decode_resize_sync(data):
    """Run in thread: decode JPEG and resize to pipeline resolution."""
    frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return None
    return cv2.resize(frame, (STREAM_WIDTH, STREAM_HEIGHT))


def _draw_encode_sync(
    frame,
    ear,
    mar,
    perclos,
    blink_count,
    head_prediction,
    eye_prediction,
    driver_state,
    driver_identity,
    alert_type,
    alert_message,
    landmarks=None,
    img_w=None,
    img_h=None,
    pitch=0.0,
    yaw=0.0,
    roll=0.0,
):
    """Run in thread: draw HUD and JPEG-encode so event loop isn't blocked."""
    draw_driver_hud(
        frame,
        ear=ear,
        mar=mar,
        perclos=perclos,
        blink_count=blink_count,
        head_prediction=head_prediction,
        eye_prediction=eye_prediction,
        driver_state=driver_state,
        driver_identity=driver_identity,
        alert_type=alert_type,
        alert_message=alert_message,
        landmarks=landmarks,
        img_w=img_w,
        img_h=img_h,
        pitch=pitch,
        yaw=yaw,
        roll=roll,
    )
    ok, buffer = cv2.imencode(
        ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), STREAM_JPEG_QUALITY]
    )
    return buffer.tobytes() if ok else None


def _do_face_recognition(frame_copy, driver_id_param, result_container):
    """Run in thread: get embedding, match driver, write into result_container."""
    if _arcface_model is None:
        return
    try:
        emb = _arcface_model.get_embedding_from_frame(frame_copy)
        if emb is None:
            return
        driver, _ = match_embedding_to_driver(emb, driver_id=driver_id_param)
        if driver is not None:
            result_container["driver_id"] = driver["driver_id"]
            result_container["display"] = f"{driver.get('name', driver['driver_id'])} ({driver['driver_id']})"
        else:
            result_container["driver_id"] = None
            result_container["display"] = "Unknown"
    except Exception as e:
        print(f"[stream] face recognition error: {e}")


def _do_insert_alert(driver_id, session_id, alert_type, confidence_score):
    """Run in thread: write one alert to MongoDB."""
    try:
        insert_alert(
            driver_id=driver_id,
            session_id=session_id,
            alert_type=alert_type,
            confidence_score=confidence_score,
            gps_latitude=None,
            gps_longitude=None,
        )
    except Exception as e:
        print(f"[stream] insert_alert error: {e}")


@router.websocket("/stream")
async def stream(websocket: WebSocket):
    """
    WebSocket endpoint for real-time driver monitoring.

    Protocol:
    - Client sends raw JPEG bytes (single frame) per message.
    - Server runs fatigue / head pose / eye gaze models,
      fuses their outputs, stores alerts in MongoDB, and
      sends back the annotated JPEG.

    Driver identity:
    - Optional `driver_id` can be passed as a query parameter on the URL:
      ws://<host>:5000/stream?driver_id=DRIVER123
    - If omitted, face recognition (ArcFace + MongoDB) runs on the stream to
      identify the driver; recognized driver_id is used for events/sessions/alerts.
    """
    await websocket.accept()
    print("WebSocket client connected")

    driver_id = websocket.query_params.get("driver_id")
    session_id: str | None = None
    # Recognized driver from face (background thread updates this)
    recognition_result: dict = {"driver_id": None, "display": "—"}

    # Throttle face recognition to every N frames to reduce CPU
    frame_count = 0
    recognition_interval = 20
    # Reuse last fusion/metrics when skipping model runs (MODEL_INTERVAL > 1)
    last_fusion_result = None
    last_hud = None  # dict of HUD params for draw_driver_hud

    try:
        loop = asyncio.get_running_loop()
        while True:
            data = await websocket.receive_bytes()

            # Decode + resize in executor so event loop isn't blocked by JPEG decode
            frame = await loop.run_in_executor(_executor, _decode_resize_sync, data)
            if frame is None:
                continue

            run_models_this_frame = (frame_count % MODEL_INTERVAL) == 0

            # 1. Face landmarks every frame (for overlay drawing)
            landmarks, img_w, img_h = await loop.run_in_executor(
                _executor, _get_landmarks_sync, frame
            )
            has_landmarks = landmarks is not None and img_w is not None and img_h is not None

            if run_models_this_frame:
                # 2–5. Run fatigue, drowsiness, head pose, eye gaze in parallel
                if not has_landmarks:
                    await asyncio.gather(
                        loop.run_in_executor(_executor, _run_fatigue, frame, landmarks, img_w, img_h),
                        loop.run_in_executor(_executor, _run_drowsiness, frame, landmarks, img_w, img_h),
                    )
                else:
                    await asyncio.gather(
                        loop.run_in_executor(_executor, _run_fatigue, frame, landmarks, img_w, img_h),
                        loop.run_in_executor(_executor, _run_drowsiness, frame, landmarks, img_w, img_h),
                        loop.run_in_executor(_executor, _run_headpose, frame, landmarks, img_w, img_h),
                        loop.run_in_executor(_executor, _run_eye_gaze, frame, landmarks, img_w, img_h),
                    )

                # 5b. Face recognition in background when throttled
                if _arcface_model is not None and (driver_id is None or frame_count % recognition_interval == 0):
                    loop.run_in_executor(
                        _executor,
                        _do_face_recognition,
                        frame.copy(),
                        driver_id,
                        recognition_result,
                    )

            recognized_driver_id = recognition_result.get("driver_id")
            recognized_driver_display = recognition_result.get("display", "—")
            effective_driver_id = driver_id if driver_id is not None else recognized_driver_id
            frame_count += 1

            if run_models_this_frame:
                # 6. Build fusion inputs and fuse
                fatigue_score = 1.0 if getattr(fatigue_model, "fatigue_active", False) else 0.0
                perclos = getattr(drowsiness_model, "perclos", 0.0)
                eye_closure_duration_sec = getattr(
                    drowsiness_model, "eye_closure_duration_sec", 0.0
                )
                head_prediction = getattr(headpose_model, "last_prediction", "Forward")
                head_turned_away_sec = getattr(headpose_model, "head_turned_away_sec", 0.0)
                is_distracted = head_prediction.lower() != "forward"
                distraction_score = 1.0 if is_distracted else 0.0

                outputs = ModelOutputs(
                    perclos=perclos,
                    blink_duration_sec=0.0,
                    blink_rate_low=getattr(drowsiness_model, "blink_rate_low", False),
                    fatigue_score=fatigue_score,
                    head_turned_away_sec=head_turned_away_sec,
                    distraction_score=distraction_score,
                    eye_closure_duration_sec=eye_closure_duration_sec,
                )
                fusion_result = fusion_engine.fuse(outputs)
                last_fusion_result = fusion_result

                # 7. Alerts/session in MongoDB (background)
                if fusion_result.alert_type:
                    if effective_driver_id and session_id is None:
                        session_id = create_session(effective_driver_id)
                        print("WebSocket session started:", session_id)
                    loop.run_in_executor(
                        _executor,
                        _do_insert_alert,
                        effective_driver_id or "UNKNOWN",
                        session_id,
                        fusion_result.alert_type,
                        fusion_result.confidence_score,
                    )

                last_hud = {
                    "ear": getattr(fatigue_model, "last_ear", None),
                    "mar": getattr(fatigue_model, "last_mar", None),
                    "perclos": perclos,
                    "blink_count": getattr(drowsiness_model, "blink_count", 0),
                    "head_prediction": head_prediction,
                    "eye_prediction": getattr(eye_gaze_model, "stable_prediction", "CENTER"),
                    "driver_state": fusion_result.driver_state,
                    "driver_identity": recognized_driver_display if _arcface_model is not None else None,
                    "alert_type": fusion_result.alert_type,
                    "alert_message": fusion_result.message or (fusion_result.alert_type or "").upper(),
                    "pitch": getattr(headpose_model, "last_pitch", 0.0),
                    "yaw": getattr(headpose_model, "last_yaw", 0.0),
                    "roll": getattr(headpose_model, "last_roll", 0.0),
                }
            else:
                # Reuse last fusion and HUD params (no model run this frame)
                fusion_result = last_fusion_result
                last_hud = last_hud or {}

            # 9–10. Draw HUD + encode in executor, then send (keeps event loop free)
            hud = last_hud or {}
            payload = await loop.run_in_executor(
                _executor,
                _draw_encode_sync,
                frame,
                hud.get("ear"),
                hud.get("mar"),
                hud.get("perclos", 0.0),
                hud.get("blink_count", 0),
                hud.get("head_prediction", "—"),
                hud.get("eye_prediction", "—"),
                hud.get("driver_state", "normal"),
                hud.get("driver_identity"),
                hud.get("alert_type"),
                hud.get("alert_message", ""),
                landmarks,
                img_w,
                img_h,
                hud.get("pitch", 0.0),
                hud.get("yaw", 0.0),
                hud.get("roll", 0.0),
            )
            if payload:
                await websocket.send_bytes(payload)

    except WebSocketDisconnect:
        print("WebSocket client disconnected")
    except Exception as e:
        # Log but do not crash the server process
        print("WebSocket error:", e)
    finally:
        if session_id:
            end_session(session_id)