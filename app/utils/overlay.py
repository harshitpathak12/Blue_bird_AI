"""
HUD overlay matching Driver Monitoring System UI.
Dark blue left panel with distraction/drowsy bars, metrics, face bbox, landmarks.
"""

from typing import Any

import cv2
import numpy as np

# Layout – dark blue theme like reference screenshot
_PANEL_X = 4
_PANEL_WIDTH = 155
_LINE_HEIGHT = 18
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.48
_FONT_THICKNESS = 1
_DARK_BLUE = (25, 35, 65)
_PANEL_BG = (30, 45, 80)
_BAR_GREEN = (0, 200, 100)
_TEXT_WHITE = (255, 255, 255)
_TEXT_LIGHT = (220, 230, 255)
_LABEL_GRAY = (180, 190, 210)
_ALERT_BG = (0, 0, 90)
_ALERT_TEXT = (0, 220, 255)
_BBOX_GREEN = (0, 255, 0)
_LANDMARK_WHITE = (255, 255, 255)
_EYE_TEAL = (180, 200, 100)  # BGR teal
_BOTTOM_BAR = (20, 35, 60)

# MediaPipe eye indices for overlay
_LEFT_EYE_IDS = [33, 160, 158, 133, 153, 144]
_RIGHT_EYE_IDS = [362, 385, 387, 263, 373, 380]


def _draw_panel_bg(frame: np.ndarray, y_start: int, height: int) -> None:
    """Draw dark blue panel background."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (_PANEL_X, y_start), (_PANEL_X + _PANEL_WIDTH, y_start + height), _PANEL_BG, -1)
    cv2.addWeighted(overlay, 0.92, frame, 0.08, 0, frame)


def _draw_bar(frame: np.ndarray, x: int, y: int, width: int, height: int, value_pct: float) -> None:
    """Draw horizontal percentage bar (green fill). Caller draws % text separately."""
    cv2.rectangle(frame, (x, y), (x + width, y + height), (50, 60, 90), -1)
    fill_w = int(width * min(1.0, max(0.0, value_pct / 100.0)))
    if fill_w > 0:
        cv2.rectangle(frame, (x, y), (x + fill_w, y + height), _BAR_GREEN, -1)


def _draw_face_overlays(
    frame: np.ndarray,
    landmarks: Any,
    img_w: int,
    img_h: int,
    pitch: float,
    yaw: float,
    roll: float,
) -> None:
    """Draw green bbox, white landmark dots, teal eye circles, 3D head axis."""
    if landmarks is None:
        return
    pts = []
    for i in range(min(478, len(landmarks))):
        lm = landmarks[i]
        px, py = int(lm.x * img_w), int(lm.y * img_h)
        pts.append((px, py))
    if not pts:
        return
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    x1, x2 = max(0, min(xs) - 15), min(img_w, max(xs) + 15)
    y1, y2 = max(0, min(ys) - 15), min(img_h, max(ys) + 15)

    # Green face bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), _BBOX_GREEN, 2)

    # White landmark dots (every 3rd to avoid clutter)
    for i in range(0, len(pts), 3):
        cv2.circle(frame, pts[i], 1, _LANDMARK_WHITE, -1)

    # Teal eye circles with crosshair (center of eye)
    for eye_ids in (_LEFT_EYE_IDS, _RIGHT_EYE_IDS):
        if max(eye_ids) >= len(landmarks):
            continue
        ex = int(np.mean([landmarks[i].x * img_w for i in eye_ids]))
        ey = int(np.mean([landmarks[i].y * img_h for i in eye_ids]))
        cv2.circle(frame, (ex, ey), 18, _EYE_TEAL, 2)
        cv2.line(frame, (ex - 8, ey), (ex + 8, ey), _LANDMARK_WHITE, 1)
        cv2.line(frame, (ex, ey - 8), (ex, ey + 8), _LANDMARK_WHITE, 1)

    # 3D head axis (simplified: draw from nose tip)
    nose_idx = 1
    if nose_idx < len(landmarks):
        nx = int(landmarks[nose_idx].x * img_w)
        ny = int(landmarks[nose_idx].y * img_h)
        scale = 25
        dx = int(scale * np.sin(np.radians(yaw)))
        dy = int(scale * np.sin(np.radians(pitch)))
        cv2.arrowedLine(frame, (nx, ny), (nx + dx, ny + dy), (0, 0, 255), 2, tipLength=0.2)


def draw_driver_hud(
    frame: np.ndarray,
    *,
    ear: float | None = None,
    mar: float | None = None,
    perclos: float = 0.0,
    blink_count: int = 0,
    head_prediction: str = "—",
    eye_prediction: str = "—",
    driver_state: str = "normal",
    driver_identity: str | None = None,
    alert_type: str | None = None,
    alert_message: str = "",
    landmarks: Any = None,
    img_w: int | None = None,
    img_h: int | None = None,
    pitch: float = 0.0,
    yaw: float = 0.0,
    roll: float = 0.0,
) -> None:
    """
    Draw Driver Monitoring System HUD (dark blue panel, metrics, face overlays).
    Matches reference screenshot layout.
    """
    h, w = frame.shape[:2]

    # Distraction & drowsy percentages
    distraction_pct = 0.0 if head_prediction.lower() == "forward" else 70.0
    drowsy_pct = min(100.0, (perclos * 100) + (20.0 if driver_state in ("fatigue", "sleep") else 0))
    ear_pct = (ear * 100) if ear is not None else 0.0

    # Expression from driver_state
    expr = driver_state.upper() if driver_state else "NEUTRAL"
    if expr not in ("NORMAL", "FATIGUE", "DISTRACTION", "SLEEP"):
        expr = "NEUTRAL"

    # Head/Gaze zone mapping
    head_zone = "FRONT" if head_prediction.lower() == "forward" else head_prediction.upper()
    gaze_zone = eye_prediction.upper() if eye_prediction else "CENTER"

    # Panel height estimate
    panel_h = 280
    _draw_panel_bg(frame, 0, panel_h)

    y = 14
    # Driver name
    name = (driver_identity or "Driver").split("(")[0].strip() or "Unknown"
    cv2.putText(frame, name[:16], (_PANEL_X + 6, y), _FONT, 0.55, _TEXT_LIGHT, 1, cv2.LINE_AA)
    y += _LINE_HEIGHT + 6

    # Distraction bar
    cv2.putText(frame, "DISTRACTION LEVEL", (_PANEL_X + 6, y), _FONT, 0.38, _LABEL_GRAY, 1, cv2.LINE_AA)
    y += 12
    _draw_bar(frame, _PANEL_X + 6, y, _PANEL_WIDTH - 42, 14, distraction_pct)
    cv2.putText(frame, f"{distraction_pct:.0f}%", (_PANEL_X + _PANEL_WIDTH - 34, y + 11), _FONT, 0.4, _TEXT_WHITE, 1, cv2.LINE_AA)
    y += 22

    # Drowsy bar
    cv2.putText(frame, "DROWSY LEVEL", (_PANEL_X + 6, y), _FONT, 0.38, _LABEL_GRAY, 1, cv2.LINE_AA)
    y += 12
    _draw_bar(frame, _PANEL_X + 6, y, _PANEL_WIDTH - 42, 14, drowsy_pct)
    cv2.putText(frame, f"{drowsy_pct:.0f}%", (_PANEL_X + _PANEL_WIDTH - 34, y + 11), _FONT, 0.4, _TEXT_WHITE, 1, cv2.LINE_AA)
    y += 26

    # Expression
    cv2.putText(frame, f"Expression: {expr}", (_PANEL_X + 6, y), _FONT, _FONT_SCALE, _TEXT_WHITE, 1, cv2.LINE_AA)
    y += _LINE_HEIGHT

    # Eye Openness
    cv2.putText(frame, f"Eye Openness: {ear_pct:.0f}% {ear_pct:.0f}%", (_PANEL_X + 6, y), _FONT, _FONT_SCALE, _TEXT_WHITE, 1, cv2.LINE_AA)
    y += _LINE_HEIGHT

    # Blink
    cv2.putText(frame, f"Blinks: {blink_count}", (_PANEL_X + 6, y), _FONT, _FONT_SCALE, _TEXT_WHITE, 1, cv2.LINE_AA)
    y += _LINE_HEIGHT + 4

    # HEAD DIR (PYR)
    cv2.putText(frame, f"HEAD DIR: {pitch:+.0f} {yaw:+.0f} {roll:+.0f}", (_PANEL_X + 6, y), _FONT, 0.4, _TEXT_WHITE, 1, cv2.LINE_AA)
    y += _LINE_HEIGHT

    # GAZE DIR / ZONE
    cv2.putText(frame, f"GAZE ZONE: {gaze_zone}", (_PANEL_X + 6, y), _FONT, 0.4, _TEXT_WHITE, 1, cv2.LINE_AA)
    y += _LINE_HEIGHT
    cv2.putText(frame, f"HEAD ZONE: {head_zone}", (_PANEL_X + 6, y), _FONT, 0.4, _TEXT_WHITE, 1, cv2.LINE_AA)

    # Face overlays on video
    if landmarks is not None and img_w is not None and img_h is not None:
        _draw_face_overlays(frame, landmarks, img_w, img_h, pitch, yaw, roll)

    # Bottom bar
    bar_h = 28
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), _BOTTOM_BAR, -1)
    cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)
    cv2.putText(frame, "Driver Monitoring System", (w - 220, h - 8), _FONT, 0.5, _TEXT_WHITE, 1, cv2.LINE_AA)
    cv2.rectangle(frame, (12, h - bar_h + 4), (140, h - 6), (80, 100, 140), 2)
    cv2.putText(frame, "Driver enrollment", (24, h - 12), _FONT, 0.45, _TEXT_WHITE, 1, cv2.LINE_AA)

    # Info callout (bottom right)
    callout = "The system infers several attributes for driving safely."
    tw, th = cv2.getTextSize(callout, _FONT, 0.4, 1)[0]
    cx, cy = w - tw - 24, h - bar_h - th - 12
    cv2.rectangle(frame, (cx - 8, cy - 4), (cx + tw + 12, cy + th + 8), (40, 55, 90), -1)
    cv2.rectangle(frame, (cx - 8, cy - 4), (cx + tw + 12, cy + th + 8), _TEXT_LIGHT, 1)
    cv2.putText(frame, callout, (cx, cy + th), _FONT, 0.4, _TEXT_WHITE, 1, cv2.LINE_AA)

    # Alert banner when fusion raised an alert
    if alert_type and alert_message:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 36), _ALERT_BG, -1)
        cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)
        cv2.putText(frame, f"  {alert_message}", (12, 24), _FONT, 0.6, _ALERT_TEXT, 2, cv2.LINE_AA)
