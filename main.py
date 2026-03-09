"""
Driver Safety System – Complete Pipeline (per PDF).
APIs: /api/login, /api/monitor, /api/alerts, /api/safety-score + sessions.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import alerts, login, monitor, safety_score, sessions, websocket

app = FastAPI(
    title="Driver Safety System",
    description="Complete pipeline: data ingestion, fusion, alerts, safety scoring (per DRIVER SAFETY SYSTEM – COMPLETE PIPELINE DESIGN)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(login.router)
app.include_router(sessions.router)
app.include_router(monitor.router)
app.include_router(alerts.router)
app.include_router(safety_score.router)
app.include_router(websocket.router)


@app.get("/")
def root():
    return {
        "service": "Driver Safety System",
        "apis": [
            "POST /api/login/",
            "POST /api/login/register",
            "POST /api/sessions/start",
            "POST /api/sessions/end",
            "POST /api/monitor/frame",
            "POST /api/alerts/",
            "GET /api/alerts/",
            "GET /api/safety-score/",
            "POST /api/safety-score/compute",
        ],
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)
