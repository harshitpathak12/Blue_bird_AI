# Driver Safety System – API & Client Documentation

**Version:** 1.0.0  
**Last Updated:** March 2025  
**Audience:** Frontend Team, Client Stakeholders

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Base URL & Environment](#2-base-url--environment)
3. [Authentication & Driver Identity](#3-authentication--driver-identity)
4. [REST API Reference](#4-rest-api-reference)
5. [WebSocket Real-Time Stream](#5-websocket-real-time-stream)
6. [Data Models & Schemas](#6-data-models--schemas)
7. [Integration Guidelines for Frontend](#7-integration-guidelines-for-frontend)
8. [Error Handling](#8-error-handling)
9. [CORS & Security](#9-cors--security)

---

## 1. Project Overview

The **Driver Safety System** is a real-time monitoring backend that:

- Detects driver fatigue, distraction, and drowsiness using computer vision
- Recognizes drivers via face authentication (ArcFace)
- Stores alerts and sessions in MongoDB
- Computes daily safety scores for fleet management

### Key Capabilities

| Feature | Description |
|---------|-------------|
| **Face Login** | Driver logs in by uploading a face image; system matches against registered drivers |
| **Face Registration** | New drivers register with name, age, and **live photo from camera** (base64, no file upload); receive a unique 5-digit `driver_id` |
| **Real-Time Stream** | WebSocket accepts video frames, runs ML models (fatigue, head pose, eye gaze), returns annotated frame with HUD |
| **Face Recognition in Stream** | When `driver_id` is not passed in the WebSocket URL, system auto-identifies the driver from the video |
| **Sessions** | Start/end driving sessions; alerts are linked to sessions |
| **Alerts** | Fatigue, distraction, sleep events stored with `driver_id`, `session_id`, timestamp, GPS (optional) |
| **Safety Score** | Daily score (0–100) derived from fatigue, distraction, sleep counts; risk levels: Safe / Moderate Risk / High Risk |

---

## 2. Base URL & Environment

| Environment | Base URL |
|-------------|----------|
| Local development | `http://localhost:5000` |
| Production | Configure per deployment (e.g. `https://api.yourdomain.com`) |

### Run the Server

```bash
cd driver_safety_system
python main.py
# OR: uvicorn main:app --host 0.0.0.0 --port 5000
```

- **Health check:** `GET /health` → `{"status": "ok"}`
- **API docs (Swagger):** `GET /docs`
- **ReDoc:** `GET /redoc` (if enabled)

---

## 3. Authentication & Driver Identity

- There is **no JWT/OAuth**; driver identity is established via **face recognition**.
- **`driver_id`** is a unique 5-digit numeric ID (e.g. `"12345"`) assigned at registration.
- Face embeddings are stored in MongoDB; login/stream use them to match drivers.

---

## 4. REST API Reference

### 4.1 Login – Face Recognition

**`POST /api/login/`**

Log in using a face image. If `driver_id` is provided, verifies against that driver; otherwise matches against all registered drivers.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `image` | File (multipart) | Yes | Face image (JPEG/PNG) |
| `driver_id` | Form | No | If provided, verify face against this driver only |

**Request:** `multipart/form-data` with `image` (and optionally `driver_id`)

**Response (200):**

```json
{
  "driver_id": "12345",
  "driver_name": "John Doe",
  "age": 35,
  "message": "Login successful"
}
```

**Errors:**

- `400` – Empty image or no face detected
- `401` – Face not recognized or invalid driver_id
- `500` – Face model not available

---

### 4.2 Register – New Driver (live photo only)

**`POST /api/login/register`**

Register a new driver using a **live photo from the camera** (no file upload). Send name, optional age, and base64-encoded image from camera capture. Returns a unique 5-digit `driver_id`.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Driver name |
| `age` | number | No | Driver age (1–120) |
| `image_base64` | string | Yes | Base64-encoded JPEG/PNG from live camera (e.g. canvas.toDataURL('image/jpeg').split(',')[1]) |

**Request:** `application/json` body:

```json
{
  "name": "Jane Smith",
  "age": 28,
  "image_base64": "<base64 string from camera capture>"
}
```

**Response (200):**

```json
{
  "driver_id": "45678",
  "driver_name": "Jane Smith",
  "age": 28,
  "message": "Registration successful"
}
```

**Errors:**

- `400` – Empty image or no face detected

---

### 4.3 Sessions

**`POST /api/sessions/start`**

Start a driving session.

**Body:**

```json
{
  "driver_id": "12345"
}
```

**Response (200):**

```json
{
  "session_id": "abc123...",
  "driver_id": "12345",
  "start_time": "2025-03-05T10:00:00Z",
  "status": "active",
  "message": "Session started"
}
```

---

**`POST /api/sessions/end`**

End a driving session.

**Body:**

```json
{
  "session_id": "abc123..."
}
```

**Response (200):**

```json
{
  "session_id": "abc123...",
  "end_time": "2025-03-05T12:00:00Z",
  "status": "ended",
  "message": "Session ended"
}
```

**Errors:**

- `404` – Session not found

---

### 4.4 Monitor – Single Frame (REST)

**`POST /api/monitor/frame`**

Ingest a single frame with metadata. Useful for testing; for live streaming, prefer WebSocket.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `driver_id` | Form | Yes | Driver ID |
| `session_id` | Form | No | Session ID |
| `gps_latitude` | Form | No | Latitude |
| `gps_longitude` | Form | No | Longitude |
| `frame` | File (multipart) | Yes | Image frame (JPEG/PNG) |

**Response (200):**

```json
{
  "processed": true,
  "alert": null,
  "driver_state": "normal"
}
```

If an alert is raised:

```json
{
  "processed": true,
  "alert": {
    "driver_id": "12345",
    "session_id": "abc123",
    "alert_type": "fatigue",
    "confidence_score": 0.85,
    "timestamp": "2025-03-05T10:30:00Z",
    "gps": { "latitude": 40.7, "longitude": -74.0 }
  },
  "driver_state": "fatigue"
}
```

---

### 4.5 Alerts

**`POST /api/alerts/`**

Manually create an alert (e.g. from external systems).

**Body:**

```json
{
  "driver_id": "12345",
  "session_id": "abc123",
  "alert_type": "fatigue",
  "confidence_score": 0.8,
  "gps": { "latitude": 40.7, "longitude": -74.0 }
}
```

**`alert_type`:** `"fatigue"` | `"distraction"` | `"sleep"`  
**`confidence_score`:** 0.0–1.0

---

**`GET /api/alerts/`**

List alerts with optional filters.

| Query Param | Type | Default | Description |
|-------------|------|---------|-------------|
| `driver_id` | string | - | Filter by driver |
| `session_id` | string | - | Filter by session |
| `limit` | int | 100 | Max alerts to return (1–500) |

**Response (200):**

```json
{
  "alerts": [
    {
      "alert_id": "...",
      "driver_id": "12345",
      "session_id": "abc123",
      "alert_type": "fatigue",
      "confidence_score": 0.85,
      "timestamp": "2025-03-05T10:30:00Z",
      "gps": { "latitude": 40.7, "longitude": -74.0 }
    }
  ],
  "total": 1
}
```

---

### 4.6 Safety Score

**`GET /api/safety-score/`**

Get daily safety scores for a driver.

| Query Param | Type | Required | Description |
|-------------|------|----------|-------------|
| `driver_id` | string | Yes | Driver ID |
| `date_from` | string | No | Start date (YYYY-MM-DD) |
| `date_to` | string | No | End date (YYYY-MM-DD) |

**Response (200):**

```json
{
  "driver_id": "12345",
  "daily_scores": [
    {
      "driver_id": "12345",
      "date": "2025-03-05",
      "fatigue_count": 2,
      "distraction_count": 1,
      "sleep_count": 0,
      "safety_score": 87.0,
      "risk_level": "Moderate Risk"
    }
  ]
}
```

**Risk levels:**

- **90–100** → Safe  
- **70–89** → Moderate Risk  
- **&lt;70** → High Risk  

**Formula:** `score = 100 - (fatigue × 5) - (distraction × 3) - (sleep × 10)`

---

**`POST /api/safety-score/compute`**

Compute and store daily score for a specific date.

| Query Param | Type | Description |
|-------------|------|-------------|
| `driver_id` | string | Driver ID |
| `score_date` | string | Date (YYYY-MM-DD) |

**Response (200):**

```json
{
  "driver_id": "12345",
  "date": "2025-03-05",
  "fatigue_count": 2,
  "distraction_count": 1,
  "sleep_count": 0,
  "safety_score": 87.0,
  "risk_level": "Moderate Risk"
}
```

---

## 5. WebSocket Real-Time Stream

**Endpoint:** `WS /stream`

**URL:** `ws://<host>:5000/stream` or `wss://<host>/stream`

### Query Parameters

| Param | Required | Description |
|-------|----------|-------------|
| `driver_id` | No | If provided, use this driver for events/sessions/alerts. If omitted, face recognition identifies the driver from the video. |

**Examples:**

- `ws://localhost:5000/stream` – Auto-identify driver from video
- `ws://localhost:5000/stream?driver_id=12345` – Use known driver

### Protocol

1. **Client** → **Server:** Send raw **JPEG bytes** (single frame) per message.
2. **Server** runs:
   - Face detection (landmarks)
   - Fatigue (EAR/MAR)
   - Drowsiness (PERCLOS, blinks)
   - Head pose (distraction)
   - Eye gaze
   - Face recognition (every 10 frames when `driver_id` not in URL)
3. **Server** → **Client:** Send back **JPEG bytes** of the annotated frame (with HUD overlay).

### Frame Overlay (HUD)

Each returned frame includes a left-side panel with:

- **Driver** – Recognized driver (e.g. `"John (12345)"`) or `"Unknown"`
- **EAR** – Eye Aspect Ratio (fatigue)
- **MAR** – Mouth Aspect Ratio (fatigue)
- **PERCLOS** – % eye closure over time
- **Blinks** – Blink count
- **Head** – Head pose (e.g. Forward, Left, Right)
- **Gaze** – Eye gaze (e.g. CENTER)
- **State** – `normal` | `fatigue` | `distraction` | `sleep`

When an alert is raised, a banner appears at the bottom (e.g. `FATIGUE`, `DISTRACTION`, `SLEEP`).

### Frontend Integration Example

```javascript
const ws = new WebSocket('ws://localhost:5000/stream?driver_id=12345');

ws.binaryType = 'arraybuffer';

ws.onopen = () => {
  // Capture webcam frame as JPEG and send
  const blob = await captureFrameAsJpeg(); // your capture logic
  ws.send(blob);
};

ws.onmessage = (event) => {
  const blob = event.data; // JPEG bytes
  const img = document.getElementById('preview');
  img.src = URL.createObjectURL(new Blob([blob]));
  // Optionally capture next frame and send (e.g. requestAnimationFrame loop)
};
```

**Recommendation:** Capture at ~10–15 FPS to balance latency and server load.

---

## 6. Data Models & Schemas

### Driver

| Field | Type | Description |
|-------|------|-------------|
| driver_id | string | Unique 5-digit ID |
| name | string | Driver name |
| age | int? | Optional age |
| face_embedding | float[] | Stored for recognition |
| face_image_path | string? | Optional path to stored image |
| created_at | datetime | Registration time |
| last_seen | datetime? | Last login/activity |

### Session

| Field | Type | Description |
|-------|------|-------------|
| session_id | string | Unique ID |
| driver_id | string | Driver ID |
| start_time | datetime | Session start |
| end_time | datetime? | Session end (null if active) |

### Alert

| Field | Type | Description |
|-------|------|-------------|
| alert_id | string | Mongo ObjectId |
| driver_id | string | Driver ID |
| session_id | string? | Session ID |
| alert_type | string | fatigue \| distraction \| sleep |
| confidence_score | float | 0.0–1.0 |
| timestamp | datetime | When alert occurred |
| gps | object? | { latitude, longitude } |

### Daily Score

| Field | Type | Description |
|-------|------|-------------|
| driver_id | string | Driver ID |
| date | string | YYYY-MM-DD |
| fatigue_count | int | Fatigue alerts that day |
| distraction_count | int | Distraction alerts |
| sleep_count | int | Sleep alerts |
| safety_score | float | 0–100 |
| risk_level | string | Safe \| Moderate Risk \| High Risk |

---

## 7. Integration Guidelines for Frontend

### Recommended Flow

1. **Login:** `POST /api/login/` with face image → get `driver_id`, `driver_name`, `age`.
2. **Start session:** `POST /api/sessions/start` with `driver_id`.
3. **Open WebSocket:** `ws://host/stream?driver_id=<driver_id>`.
4. **Send frames:** Capture camera, encode as JPEG, send bytes.
5. **Display annotated frame:** Receive JPEG, render in `<img>` or `<video>` element.
6. **End session:** `POST /api/sessions/end` with `session_id` when trip ends.
7. **Dashboard:** `GET /api/alerts/?driver_id=...` and `GET /api/safety-score/?driver_id=...`.

### Frame Capture (Browser)

```javascript
// Example: capture from <video> (webcam stream)
async function captureFrameAsJpeg(videoElement, quality = 0.8) {
  const canvas = document.createElement('canvas');
  canvas.width = 640;
  canvas.height = 480;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(videoElement, 0, 0);
  return await new Promise(resolve => {
    canvas.toBlob(resolve, 'image/jpeg', quality);
  });
}
```

### CORS

The backend allows all origins (`allow_origins=["*"]`). For production, restrict to your frontend domain.

---

## 8. Error Handling

- REST APIs return standard HTTP status codes (400, 401, 404, 500).
- JSON error body: `{"detail": "Message"}` (FastAPI default).
- WebSocket: On error, connection may close; check `onerror` and `onclose`.

---

## 9. CORS & Security

- **CORS:** Enabled for all origins in development.
- **MongoDB:** Connection string in `configs/config.yaml`; database: `driver_monitoring`.
- **Secrets:** Do not commit MongoDB credentials; use environment variables in production.

---

## Quick Reference – All Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/` | Service info + API list |
| GET | `/health` | Health check |
| GET | `/docs` | Swagger UI |
| POST | `/api/login/` | Face login |
| POST | `/api/login/register` | Face registration |
| POST | `/api/sessions/start` | Start session |
| POST | `/api/sessions/end` | End session |
| POST | `/api/monitor/frame` | Single frame (REST) |
| POST | `/api/alerts/` | Create alert |
| GET | `/api/alerts/` | List alerts |
| GET | `/api/safety-score/` | Get daily scores |
| POST | `/api/safety-score/compute` | Compute & store daily score |
| WS | `/stream` | Real-time video stream |

---

**Contact:** For technical questions, reach out to the backend/ML team.
