"""
Modern Driver Monitoring System HUD
Automotive-style interface inspired by Qualcomm demo UI
"""

from typing import Any, Tuple
import cv2
import numpy as np


# ----------------------------
# UI STYLE
# ----------------------------

_PANEL_X = 10
_PANEL_WIDTH = 270
_LINE_HEIGHT = 26

_FONT = cv2.FONT_HERSHEY_SIMPLEX

_TEXT_WHITE = (240, 245, 250)
_TEXT_LIGHT = (200, 210, 225)
_LABEL_GRAY = (140, 150, 170)

_PANEL_BG = (70, 50, 35)  # Dark blue (BGR)
_BAR_BG = (95, 75, 55)
_BAR_FILL = (0, 210, 210)

_BBOX_COLOR = (0, 230, 200)
_LANDMARK_COLOR = (220, 230, 240)

_BOTTOM_BAR = (65, 50, 40)  # Blue

_AXIS_RED = (90, 90, 255)
_AXIS_GREEN = (90, 255, 140)
_AXIS_BLUE = (255, 180, 80)

_ALERT_BG = (65, 50, 40) # Blue
_ALERT_TEXT = (100, 220, 255)


# ----------------------------
# MEDIAPIPE LANDMARK IDS
# ----------------------------

_LEFT_EYE_IDS = [33,160,158,133,153,144]
_RIGHT_EYE_IDS = [362,385,387,263,373,380]

_KEY_LANDMARK_IDS = [
1,2,61,291,0,17,
33,133,159,145,
362,263,386,374,
10,152,234,454
]

_NOSE_TIP_IDX = 1


# ----------------------------
# UTILS
# ----------------------------

def draw_rounded_rect(img, x, y, w, h, r, color, thickness=-1):

    if thickness < 0:

        cv2.rectangle(img,(x+r,y),(x+w-r,y+h),color,-1)
        cv2.rectangle(img,(x,y+r),(x+w,y+h-r),color,-1)

        cv2.circle(img,(x+r,y+r),r,color,-1)
        cv2.circle(img,(x+w-r,y+r),r,color,-1)
        cv2.circle(img,(x+r,y+h-r),r,color,-1)
        cv2.circle(img,(x+w-r,y+h-r),r,color,-1)

    else:

        cv2.rectangle(img,(x+r,y),(x+w-r,y+h),color,thickness)
        cv2.rectangle(img,(x,y+r),(x+w,y+h-r),color,thickness)


def draw_text(frame,text,x,y,scale,color):

    cv2.putText(frame,text,(x+1,y+1),_FONT,scale,(0,0,0),3,cv2.LINE_AA)
    cv2.putText(frame,text,(x,y),_FONT,scale,color,1,cv2.LINE_AA)


def divider(frame, y):

    cv2.line(
        frame,
        (_PANEL_X + 10, y),
        (_PANEL_X + _PANEL_WIDTH - 10, y),
        (120, 100, 80),  # Blue-tinted (BGR)
        1,
    )


# ----------------------------
# PANEL
# ----------------------------

def _draw_panel_bg(frame,height):

    overlay = frame.copy()

    draw_rounded_rect(
        overlay,
        _PANEL_X,
        0,
        _PANEL_WIDTH,
        height,
        18,
        _PANEL_BG,
        -1
    )

    cv2.addWeighted(overlay,0.93,frame,0.07,0,frame)

    cv2.line(
        frame,
        (_PANEL_X+_PANEL_WIDTH-3,10),
        (_PANEL_X+_PANEL_WIDTH-3,height-10),
        (0,220,220),
        2
    )


# ----------------------------
# PROGRESS BAR
# ----------------------------

def _draw_bar(frame,x,y,width,height,value_pct):

    draw_rounded_rect(frame,x,y,width,height,6,_BAR_BG,-1)

    fill = int(width * min(1.0, max(0.0, value_pct / 100)))

    if fill > 0:
        r = min(6, fill // 2, height // 2)
        if r > 0:
            draw_rounded_rect(frame, x, y, fill, height, r, _BAR_FILL, -1)
        else:
            cv2.rectangle(frame, (x, y), (x + fill, y + height), _BAR_FILL, -1)


# ----------------------------
# ROTATION MATRIX
# ----------------------------

def _rotation_matrix(p,y,r):

    p,y,r = np.radians([p,y,r])

    cx,sx = np.cos(p),np.sin(p)
    cy,sy = np.cos(y),np.sin(y)
    cz,sz = np.cos(r),np.sin(r)

    Rx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])
    Ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
    Rz = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])

    return Rz @ Ry @ Rx


# ----------------------------
# MINI 3D AXIS
# ----------------------------

def _draw_mini_axis(frame, w, pitch, yaw, roll):

    # Top-right mini 3D axis (below name/ID)
    ox, oy = w - 70, 72
    scale = 26

    R = _rotation_matrix(pitch, yaw, roll)

    axes = [
        np.array([1, 0, 0]),
        np.array([0, 1, 0]),
        np.array([0, 0, 1]),
    ]

    colors = [_AXIS_RED, _AXIS_GREEN, _AXIS_BLUE]

    for axis, color in zip(axes, colors):

        d = R @ axis

        dx = int(d[0] * scale)
        dy = int(d[1] * scale)

        cv2.line(frame, (ox, oy), (ox + dx, oy + dy), color, 3)


# ----------------------------
# FACE OVERLAYS
# ----------------------------

def draw_face_overlay(frame, landmarks, img_w, img_h):
    """Public alias used by websocket.py."""
    return _draw_face_overlays(frame, landmarks, img_w, img_h)


def _draw_face_overlays(frame, landmarks, img_w, img_h):

    if landmarks is None or len(landmarks) < 5:
        return

    pts = []

    for lm in landmarks:

        x=int(lm.x*img_w)
        y=int(lm.y*img_h)

        pts.append((x,y))

    if not pts:
        return

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]

    pad = 20

    x1 = max(0, min(xs) - pad)
    x2 = min(img_w, max(xs) + pad)
    y1 = max(0, min(ys) - pad)
    y2 = min(img_h, max(ys) + pad)

    # glowing bbox
    for i in range(3):
        cv2.rectangle(
            frame,
            (x1-i,y1-i),
            (x2+i,y2+i),
            _BBOX_COLOR,
            1
        )

    # landmarks
    for i in _KEY_LANDMARK_IDS:

        if i < len(pts):

            cv2.circle(frame,pts[i],2,_LANDMARK_COLOR,-1)

    # eyes
    for eye_ids in (_LEFT_EYE_IDS, _RIGHT_EYE_IDS):
        if max(eye_ids) >= len(landmarks):
            continue
        ex = int(np.mean([landmarks[i].x * img_w for i in eye_ids]))
        ey = int(np.mean([landmarks[i].y * img_h for i in eye_ids]))

        cv2.circle(frame,(ex,ey),12,(200,180,100),2)

        cv2.line(frame,(ex-5,ey),(ex+5,ey),_LANDMARK_COLOR,1)
        cv2.line(frame,(ex,ey-5),(ex,ey+5),_LANDMARK_COLOR,1)


# ----------------------------
# MAIN HUD
# ----------------------------

def draw_driver_hud(
    frame,
    ear=None,
    mar=None,
    perclos=0,
    blink_count=0,
    blink_rate_hz=0,
    head_prediction="forward",
    eye_prediction="center",
    driver_state="normal",
    driver_identity=None,
    alert_type=None,
    alert_message="",
    landmarks=None,
    img_w=None,
    img_h=None,
    pitch=0,
    yaw=0,
    roll=0,
    **kwargs,
):

    h, w = frame.shape[:2]

    head = (head_prediction or "forward").lower()
    distraction = 0 if head == "forward" else 70
    drowsy = min(100, perclos * 100)

    ear_pct = (ear * 100) if ear else 0

    # panel
    _draw_panel_bg(frame, h)

    y = 30

    draw_text(frame, "DISTRACTION LEVEL", _PANEL_X + 15, y, 0.55, _LABEL_GRAY)

    y+=20

    _draw_bar(frame,_PANEL_X+15,y,170,18,distraction)

    draw_text(frame,f"{distraction}%",_PANEL_X+200,y+14,0.55,_TEXT_WHITE)

    y+=40

    draw_text(frame,"DROWSY LEVEL",_PANEL_X+15,y,0.55,_LABEL_GRAY)

    y+=20

    _draw_bar(frame,_PANEL_X+15,y,170,18,drowsy)

    draw_text(frame,f"{drowsy}%",_PANEL_X+200,y+14,0.55,_TEXT_WHITE)

    y+=40

    divider(frame,y)
    y+=20

    draw_text(frame,f"EYE OPENNESS: {ear_pct:.0f}%",_PANEL_X+15,y,0.6,_TEXT_WHITE)

    y+=30

    draw_text(frame,f"BLINK RATE: {blink_rate_hz:.2f}/s",_PANEL_X+15,y,0.6,_TEXT_WHITE)

    y+=30

    draw_text(frame,f"HEAD DIR: {pitch:+.0f} {yaw:+.0f} {roll:+.0f}",_PANEL_X+15,y,0.6,_TEXT_WHITE)

    # face overlays
    if landmarks is not None and img_w and img_h:
        _draw_face_overlays(frame, landmarks, img_w, img_h)

    # Name + Register ID at top-right
    raw = (driver_identity or "Driver").strip()
    if "(" in raw and ")" in raw:
        name = raw.split("(")[0].strip() or "Unknown"
        reg_id = raw.split("(")[1].split(")")[0].strip()
    else:
        name = raw or "Unknown"
        reg_id = ""
    name_short = name[:18] if name else "Unknown"
    tw1, _ = cv2.getTextSize(name_short, _FONT, 0.72, 2)[0]
    tw2, _ = cv2.getTextSize(reg_id or "—", _FONT, 0.55, 2)[0]
    rx = w - max(tw1, tw2) - 16
    draw_text(frame, name_short, rx, 24, 0.72, _TEXT_LIGHT)
    draw_text(frame, reg_id or "—", rx, 46, 0.55, _LABEL_GRAY)

    # axis (below name/ID)
    _draw_mini_axis(frame, w, pitch, yaw, roll)

    # bottom bar
    bar_h=45

    overlay=frame.copy()

    draw_rounded_rect(
        overlay,
        10,
        h-bar_h,
        w-20,
        bar_h-5,
        12,
        _BOTTOM_BAR,
        -1
    )

    cv2.addWeighted(overlay,0.92,frame,0.08,0,frame)

    draw_text(
        frame,
        "Driver Monitoring System",
        w // 2 - 160,
        h - 15,
        0.7,
        _TEXT_WHITE,
    )

    # Alert banner (max 3 words)
    if alert_type:
        short = {"fatigue": "FATIGUE", "distraction": "DISTRACTION", "sleep": "SLEEP"}.get(
            (alert_type or "").lower(), (alert_type or "").upper()[:12]
        )
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 40), _ALERT_BG, -1)
        cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)
        cv2.putText(frame, f"  {short}", (16, 28), _FONT, 0.75, _ALERT_TEXT, 2, cv2.LINE_AA)