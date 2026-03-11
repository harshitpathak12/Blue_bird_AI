import asyncio
import json
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.fusion import FusionEngine, ModelOutputs
from app.services.driver_identity import match_embedding_to_driver
from database.session_repository import create_session, end_session
from database.alert_repository import insert_alert

from models.face_detection.face_detection import FaceDetector
from models.fatigue_detection.fatigue_detection_model import ModelFatigue
from models.distraction_detection.model import ModelHeadPose
from models.blink_perclos.model import ModelEyeGaze
from models.blink_perclos.drowsiness_model import ModelDrowsiness

router = APIRouter()

STREAM_WIDTH, STREAM_HEIGHT = 640, 480
JPEG_SEND_QUALITY = 90
MODEL_INTERVAL = 2

_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="stream")

face_detector = FaceDetector()
fatigue_model = ModelFatigue()
headpose_model = ModelHeadPose()
eye_gaze_model = ModelEyeGaze()
drowsiness_model = ModelDrowsiness()
fusion_engine = FusionEngine()

try:
    from models.face_recongnition.face_recognition import ArcFaceModel
    _arcface_model: ArcFaceModel | None = ArcFaceModel()
except Exception as e:
    print(f"[stream] ArcFace not loaded: {e}")
    _arcface_model = None


def _process_frame(data: bytes, run_models: bool, frame_count: int,
                   driver_id, recognition_result: dict, last_metrics: dict):
    """
    Single synchronous function that runs the ENTIRE pipeline in one thread.
    Eliminates multiple run_in_executor round-trips.
    """
    frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return None, None, last_metrics

    h_in, w_in = frame.shape[:2]
    if (w_in, h_in) != (STREAM_WIDTH, STREAM_HEIGHT):
        frame = cv2.resize(frame, (STREAM_WIDTH, STREAM_HEIGHT),
                           interpolation=cv2.INTER_LINEAR)

    landmarks, img_w, img_h = face_detector.get_landmarks(frame)
    has_landmarks = landmarks is not None

    if run_models:
        if has_landmarks:
            fatigue_model.process(frame, landmarks, img_w, img_h)
            drowsiness_model.process(frame, landmarks, img_w, img_h)
            headpose_model.process(frame, landmarks, img_w, img_h)
            eye_gaze_model.process(frame, landmarks, img_w, img_h)
        else:
            fatigue_model.process(frame, None, None, None)
            drowsiness_model.process(frame, None, None, None)

        if (_arcface_model is not None
                and frame_count % 20 == 0
                and (driver_id is None or frame_count % 20 == 0)):
            try:
                emb = _arcface_model.get_embedding_from_frame(frame)
                if emb is not None:
                    driver, _ = match_embedding_to_driver(emb, driver_id=driver_id)
                    if driver is not None:
                        recognition_result["driver_id"] = driver["driver_id"]
                        recognition_result["display"] = (
                            f"{driver.get('name', driver['driver_id'])} "
                            f"({driver['driver_id']})"
                        )
                    else:
                        recognition_result["driver_id"] = None
                        recognition_result["display"] = "Unknown"
            except Exception as e:
                print(f"[stream] face recognition error: {e}")

        fatigue_score = 1.0 if fatigue_model.fatigue_active else 0.0
        perclos = drowsiness_model.perclos
        eye_closure_duration_sec = drowsiness_model.eye_closure_duration_sec
        head_prediction = headpose_model.last_prediction
        head_turned_away_sec = headpose_model.head_turned_away_sec
        distraction_score = 1.0 if head_prediction.lower() != "forward" else 0.0

        outputs = ModelOutputs(
            perclos=perclos,
            blink_duration_sec=0.0,
            blink_rate_low=drowsiness_model.blink_rate_low,
            fatigue_score=fatigue_score,
            head_turned_away_sec=head_turned_away_sec,
            distraction_score=distraction_score,
            eye_closure_duration_sec=eye_closure_duration_sec,
        )
        fusion_result = fusion_engine.fuse(outputs)

        recognized_display = recognition_result.get("display", "—")
        last_metrics = {
            "driver_state": fusion_result.driver_state,
            "alert_type": fusion_result.alert_type,
            "alert_message": fusion_result.message or "",
            "confidence_score": round(fusion_result.confidence_score, 3),

            "ear": round(fatigue_model.last_ear, 4),
            "mar": round(fatigue_model.last_mar, 4),
            "fatigue_active": fatigue_model.fatigue_active,

            "perclos": round(perclos, 4),
            "blink_count": drowsiness_model.blink_count,
            "blink_rate_hz": round(drowsiness_model.blink_rate_hz, 3),
            "blink_rate_low": drowsiness_model.blink_rate_low,
            "eye_closure_duration_sec": round(eye_closure_duration_sec, 2),

            "head_prediction": head_prediction,
            "head_turned_away_sec": round(head_turned_away_sec, 2),
            "pitch": round(headpose_model.last_pitch, 2),
            "yaw": round(headpose_model.last_yaw, 2),
            "roll": round(headpose_model.last_roll, 2),

            "eye_prediction": eye_gaze_model.stable_prediction,
            "driver_identity": recognized_display if _arcface_model is not None else None,
        }

    ok, buf = cv2.imencode(
        ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_SEND_QUALITY]
    )
    jpeg_bytes = buf.tobytes() if ok else None

    return jpeg_bytes, last_metrics.get("alert_type"), last_metrics


def _do_insert_alert(driver_id, session_id, alert_type, confidence_score):
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
    Real-time driver monitoring over WebSocket.

    Protocol:
        Client → Server:  raw JPEG bytes (one frame per message)
        Server → Client:  JSON text (metrics), then JPEG bytes (clean frame)

    Frame dropping: if multiple frames arrive while processing, only the
    latest is used — prevents latency buildup.
    """
    await websocket.accept()
    print("WebSocket client connected")

    driver_id = websocket.query_params.get("driver_id")
    session_id: str | None = None
    recognition_result: dict = {"driver_id": None, "display": "—"}

    frame_count = 0
    last_metrics: dict = {"driver_state": "waiting"}
    latest_data: bytes | None = None
    processing = False

    loop = asyncio.get_running_loop()

    async def _recv_loop():
        """Continuously receive frames; only keep the latest."""
        nonlocal latest_data
        try:
            while True:
                latest_data = await websocket.receive_bytes()
        except WebSocketDisconnect:
            pass

    recv_task = asyncio.create_task(_recv_loop())

    try:
        while not recv_task.done():
            if latest_data is None:
                await asyncio.sleep(0.005)
                continue

            data = latest_data
            latest_data = None

            run_models = (frame_count % MODEL_INTERVAL) == 0

            jpeg_bytes, alert_type, last_metrics = await loop.run_in_executor(
                _executor,
                _process_frame,
                data,
                run_models,
                frame_count,
                driver_id,
                recognition_result,
                last_metrics,
            )

            frame_count += 1

            if jpeg_bytes is None:
                continue

            if alert_type and run_models:
                effective_driver_id = driver_id or recognition_result.get("driver_id")
                if effective_driver_id and session_id is None:
                    session_id = create_session(effective_driver_id)
                    print("WebSocket session started:", session_id)
                loop.run_in_executor(
                    _executor,
                    _do_insert_alert,
                    effective_driver_id or "UNKNOWN",
                    session_id,
                    alert_type,
                    last_metrics.get("confidence_score", 0),
                )

            try:
                await websocket.send_text(json.dumps(last_metrics))
                await websocket.send_bytes(jpeg_bytes)
            except Exception:
                break

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print("WebSocket error:", e)
    finally:
        recv_task.cancel()
        print("WebSocket client disconnected")
        if session_id:
            end_session(session_id)
