# Driver Safety System

Real-time driver monitoring backend: face recognition, fatigue and distraction detection, alerts, and daily safety scoring. Built for the **Driver Safety System – Complete Pipeline** design.

## Features

- **Face authentication** – Login by face image; register with **live photo from camera only** (no file upload); unique 5-digit `driver_id` per driver
- **Real-time video stream** – WebSocket accepts camera frames, runs ML models, returns annotated frames with a HUD overlay
- **In-stream face recognition** – When `driver_id` is not provided, the system can identify the driver from the video
- **Multi-model pipeline** – Face landmarks, fatigue (EAR/MAR), drowsiness (PERCLOS, blinks), head pose, eye gaze
- **Fusion engine** – Combines model outputs into a single driver state: `normal` | `fatigue` | `distraction` | `sleep`
- **MongoDB persistence** – Alerts, sessions, drivers, daily safety scores
- **REST API** – Login, register, sessions, monitor frame, alerts, safety score (see [API docs](docs/API_AND_CLIENT_DOCUMENTATION.md))

## Tech Stack

- **Python 3.12+**
- **FastAPI** – REST + WebSocket
- **MongoDB** – Alerts, sessions, drivers, daily scores
- **OpenCV, MediaPipe** – Face detection and landmarks
- **ONNX Runtime** – ArcFace face recognition
- **RetinaFace** – Face detection for recognition pipeline
- **scikit-learn** – Head pose and eye gaze classifiers

## Prerequisites

- Python 3.12 or higher
- MongoDB (local or [MongoDB Atlas](https://www.mongodb.com/atlas))
- (Optional) Pre-trained model files in `fine_tunned_pre_train/` for full pipeline (see [Configuration](#configuration))

## Installation

```bash
# Clone the repository (or navigate to project root)
cd driver_safety_system

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate   # Linux/macOS
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Edit **`configs/config.yaml`**:

```yaml
mongodb:
  url: "mongodb+srv://USER:PASSWORD@CLUSTER.mongodb.net/"   # Your MongoDB connection string
  database: "driver_monitoring"   # Database name (default)
```

- **MongoDB URL** – Use your Atlas connection string or a local MongoDB URI. Ensure the server’s IP is allowed in Atlas Network Access if using Atlas.
- **Database** – All collections (`events`, `alerts`, `sessions`, `drivers`, `daily_scores`) are created in this database.

Optional: set **`MODEL_BASE_DIR`** to point to the directory containing model files (default: `fine_tunned_pre_train/` in the project root). Expected files include:

- `face_landmarker.task` (MediaPipe)
- `headpose_classifier.pkl`, `eye_gaze.pkl`, `eye_scaler.pkl`
- `arcface.onnx`, `arcface_db.npz` (for face recognition)

The server starts even if some model files are missing; affected features are disabled.

## Running the Server

From the project root (`driver_safety_system`):

```bash
python main.py
```

Or with uvicorn:

```bash
uvicorn main:app --host 0.0.0.0 --port 5000 --reload
```

- **API base:** `http://localhost:5000`
- **Health:** [http://localhost:5000/health](http://localhost:5000/health)
- **Swagger UI:** [http://localhost:5000/docs](http://localhost:5000/docs)
- **WebSocket stream:** `ws://localhost:5000/stream` (optional query: `?driver_id=12345`)

On startup you should see:

- `MongoDB connected: database=driver_monitoring`
- Any model load messages or warnings

If MongoDB connection fails, the process will exit with an error (see [MongoDB troubleshooting](docs/MONGODB_TROUBLESHOOTING.md)).

## Project Structure

```
driver_safety_system/
├── main.py                 # Application entry point
├── configs/
│   ├── config.yaml         # MongoDB URL and database name
│   └── config_loader.py
├── app/
│   ├── api/                # REST and WebSocket routes
│   │   ├── login.py        # Face login & register
│   │   ├── sessions.py     # Session start/end
│   │   ├── monitor.py      # Single-frame ingest
│   │   ├── alerts.py      # Alerts CRUD
│   │   ├── safety_score.py # Daily safety scores
│   │   └── websocket.py    # Real-time /stream
│   ├── fusion/             # Fusion engine (driver state, alerts)
│   ├── scoring/            # Safety score formula
│   ├── schemas/            # Pydantic request/response models
│   ├── services/           # e.g. driver identity matching
│   └── utils/              # e.g. frame overlay (HUD)
├── database/               # MongoDB collections and repositories
├── models/                 # ML models (face, fatigue, head pose, gaze, etc.)
├── docs/
│   ├── API_AND_CLIENT_DOCUMENTATION.md   # Full API reference for frontend/client
│   ├── MONGODB_TROUBLESHOOTING.md        # Why data might not appear in MongoDB
│   └── API_PAYLOADS.md
├── requirements.txt
└── README.md
```

## Documentation

| Document | Description |
|----------|-------------|
| [API and Client Documentation](docs/API_AND_CLIENT_DOCUMENTATION.md) | Full REST and WebSocket API reference for frontend and client integration |
| [MongoDB Troubleshooting](docs/MONGODB_TROUBLESHOOTING.md) | Why data might not show in MongoDB and how to fix it |

## Quick Start (WebSocket)

1. Start the server: `python main.py`
2. Connect to `ws://localhost:5000/stream` (optionally add `?driver_id=12345` if the driver is known).
3. Send JPEG frame bytes (binary) on the WebSocket.
4. Receive annotated JPEG frames with the HUD (EAR, PERCLOS, head, gaze, driver state, alerts).

For detailed request/response formats and all endpoints, see [API and Client Documentation](docs/API_AND_CLIENT_DOCUMENTATION.md).

## License

See the repository or organization for license details.
