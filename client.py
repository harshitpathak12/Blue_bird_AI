"""
WebSocket Client — Driver Safety System.

Display: Clean video on the LEFT, status panel on the RIGHT (no overlap).

Protocol:
    Client → Server:  raw JPEG bytes (one frame per message)
    Server → Client:  JSON text (metrics), then JPEG bytes (clean frame)

Usage:
    python client.py                                         # default ws://localhost:5000/stream
    python client.py --url ws://20.219.128.113:5000/stream   # remote server
    python client.py --source 1                              # alternate camera

Press 'x' to quit.
"""

import argparse
import asyncio
import json
import sys
import time

import cv2
import numpy as np
import websockets

CAPTURE_WIDTH = 640
CAPTURE_HEIGHT = 480
SEND_JPEG_QUALITY = 75
PANEL_WIDTH = 330
FONT = cv2.FONT_HERSHEY_SIMPLEX

COL_BG = (30, 30, 30)
COL_WHITE = (240, 240, 240)
COL_LIGHT = (200, 200, 200)
COL_GRAY = (130, 130, 130)
COL_GREEN = (0, 200, 100)
COL_YELLOW = (0, 210, 255)
COL_RED = (0, 70, 255)
COL_CYAN = (220, 180, 0)
COL_BAR_BG = (55, 55, 55)
COL_ALERT_BG = (0, 0, 120)

_encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), SEND_JPEG_QUALITY]


def _bar(panel, x, y, w, h, pct, color):
    cv2.rectangle(panel, (x, y), (x + w, y + h), COL_BAR_BG, -1)
    fill = int(w * max(0.0, min(1.0, pct)))
    if fill > 0:
        cv2.rectangle(panel, (x, y), (x + fill, y + h), color, -1)


def _text(panel, text, x, y, scale=0.50, color=COL_WHITE, thick=1):
    cv2.putText(panel, text, (x, y), FONT, scale, color, thick, cv2.LINE_AA)


def _divider(panel, y, w):
    cv2.line(panel, (15, y), (w - 15, y), (60, 60, 60), 1)


def _friendly_state(state):
    return {
        "NORMAL": ("All Good", COL_GREEN),
        "FATIGUE": ("Getting Tired", COL_YELLOW),
        "DISTRACTION": ("Not Looking at Road!", COL_RED),
        "SLEEP": ("Falling Asleep!", COL_RED),
        "WAITING": ("Starting...", COL_GRAY),
    }.get(state, (state, COL_WHITE))


def _friendly_head(pred):
    p = pred.upper()
    if p == "FORWARD":
        return "At the Road", COL_GREEN
    if p in ("LEFT", "RIGHT"):
        return f"Looking {pred.title()}", COL_YELLOW
    if p in ("UP", "DOWN"):
        return f"Looking {pred.title()}", COL_YELLOW
    return f"{pred.title()}", COL_YELLOW


def _friendly_eye(pred):
    p = pred.upper()
    if p == "CENTER":
        return "Focused", COL_GREEN
    return f"Looking {pred.title()}", COL_YELLOW


def _friendly_alert(alert_type):
    return {
        "fatigue": "TAKE A BREAK!",
        "distraction": "LOOK AT THE ROAD!",
        "sleep": "WAKE UP!",
    }.get((alert_type or "").lower(), "WARNING!")


def build_panel(metrics, frame_h):
    panel = np.full((frame_h, PANEL_WIDTH, 3), COL_BG, dtype=np.uint8)
    w = PANEL_WIDTH
    px = 20
    y = 32

    # Title
    _text(panel, "Driver Safety Monitor", px, y, 0.62, COL_CYAN, 2)
    y += 12
    _divider(panel, y, w)
    y += 28

    # Driver identity
    identity = metrics.get("driver_identity")
    if identity and identity != "—" and identity != "Unknown":
        _text(panel, f"Driver: {str(identity)[:22]}", px, y, 0.48, COL_LIGHT)
        y += 28

    # Driver Status — big and clear
    state = metrics.get("driver_state", "waiting").upper()
    state_text, state_color = _friendly_state(state)
    _text(panel, "Status", px, y, 0.45, COL_GRAY)
    y += 24
    _text(panel, state_text, px, y, 0.72, state_color, 2)

    y += 38
    _divider(panel, y, w)
    y += 22

    # Looking Direction
    head_pred = metrics.get("head_prediction", "?")
    head_text, head_color = _friendly_head(head_pred)
    _text(panel, "Head Direction", px, y, 0.45, COL_GRAY)
    y += 22
    _text(panel, head_text, px, y, 0.58, head_color)

    head_away = metrics.get("head_turned_away_sec", 0) or 0
    if head_away > 0.5:
        y += 22
        away_color = COL_RED if head_away >= 3.0 else COL_YELLOW
        secs = f"{head_away:.1f}"
        _text(panel, f"Not looking at road for {secs}s", px, y, 0.42, away_color)

    # Eye Focus
    y += 30
    eye_pred = metrics.get("eye_prediction", "?")
    eye_text, eye_color = _friendly_eye(eye_pred)
    _text(panel, "Eye Focus", px, y, 0.45, COL_GRAY)
    _text(panel, eye_text, px + 120, y, 0.48, eye_color)

    y += 20
    _divider(panel, y, w)
    y += 22

    # Eyes Open/Closed — simple bar
    ear = metrics.get("ear", 0) or 0
    eye_pct = min(ear / 0.35, 1.0)
    if ear > 0.25:
        eye_status, eye_col = "Open", COL_GREEN
    elif ear > 0.18:
        eye_status, eye_col = "Half Open", COL_YELLOW
    else:
        eye_status, eye_col = "Closed", COL_RED
    _text(panel, "Eyes", px, y, 0.45, COL_GRAY)
    _text(panel, eye_status, px + 80, y, 0.48, eye_col)
    y += 12
    _bar(panel, px, y, 220, 10, eye_pct, eye_col)

    # Mouth
    y += 26
    mar = metrics.get("mar", 0) or 0
    if mar > 0.5:
        mouth_status, mouth_col = "Yawning", COL_RED
    elif mar > 0.35:
        mouth_status, mouth_col = "Slightly Open", COL_YELLOW
    else:
        mouth_status, mouth_col = "Closed", COL_GREEN
    _text(panel, "Mouth", px, y, 0.45, COL_GRAY)
    _text(panel, mouth_status, px + 80, y, 0.48, mouth_col)

    # Eye Closure over time
    y += 28
    perclos = metrics.get("perclos", 0) or 0
    perc_pct = int(perclos * 100)
    if perc_pct > 40:
        perc_col = COL_RED
    elif perc_pct > 20:
        perc_col = COL_YELLOW
    else:
        perc_col = COL_GREEN
    _text(panel, "Eye Closure", px, y, 0.45, COL_GRAY)
    _text(panel, f"{perc_pct}%", px + 140, y, 0.48, perc_col)
    y += 12
    _bar(panel, px, y, 220, 10, perclos, perc_col)

    # Blink Rate
    y += 26
    blink_hz = metrics.get("blink_rate_hz", 0) or 0
    blinks = metrics.get("blink_count", 0) or 0
    _text(panel, "Blinks", px, y, 0.45, COL_GRAY)
    _text(panel, f"{blink_hz:.1f}/sec  ({blinks} total)", px + 80, y, 0.42, COL_LIGHT)

    # Fatigue warning
    y += 28
    if metrics.get("fatigue_active", False):
        _text(panel, "Driver is getting tired!", px, y, 0.52, COL_RED, 2)
        y += 26

    # Prolonged eye closure
    closure = metrics.get("eye_closure_duration_sec", 0) or 0
    if closure > 0.5:
        _text(panel, f"Eyes closed for {closure:.1f}s!", px, y, 0.48, COL_RED)
        y += 26

    # Alert banner — big, bold, clear
    alert = metrics.get("alert_type")
    if alert:
        y += 4
        _divider(panel, y, w)
        y += 8
        banner_h = 50
        cv2.rectangle(panel, (10, y), (w - 10, y + banner_h), COL_ALERT_BG, -1)
        alert_msg = _friendly_alert(alert)
        text_size = cv2.getTextSize(alert_msg, FONT, 0.72, 2)[0]
        text_x = (w - text_size[0]) // 2
        _text(panel, alert_msg, text_x, y + 33, 0.72, (255, 255, 255), 2)

    return panel


async def stream(url: str, source):
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open video source: {source}")
        sys.exit(1)

    print(f"[client] Connecting to {url}")

    async with websockets.connect(url, max_size=None) as ws:
        print("[client] Connected — streaming frames...")

        fps_time = time.time()
        fps_count = 0
        fps_display = 0.0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            _, buffer = cv2.imencode(".jpg", frame, _encode_params)
            await ws.send(buffer.tobytes())

            metrics_raw = await ws.recv()
            metrics = json.loads(metrics_raw)

            data = await ws.recv()
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

            if img is not None:
                h, w_img = img.shape[:2]
                panel = build_panel(metrics, h)

                fps_count += 1
                now = time.time()
                elapsed = now - fps_time
                if elapsed >= 1.0:
                    fps_display = fps_count / elapsed
                    fps_count = 0
                    fps_time = now

                _text(panel, f"{fps_display:.0f} FPS", PANEL_WIDTH - 85, h - 15, 0.42, COL_GRAY)

                combined = np.hstack([img, panel])
                cv2.imshow("Driver Safety System", combined)

            if cv2.waitKey(1) & 0xFF == ord("x"):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("[client] Done.")


def main():
    parser = argparse.ArgumentParser(description="Driver Safety System — WebSocket client")
    parser.add_argument("--url", default="ws://localhost:5000/stream", help="WebSocket server URL")
    parser.add_argument("--source", default="0", help="Camera index or video file path")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    asyncio.run(stream(args.url, source))


if __name__ == "__main__":
    main()
