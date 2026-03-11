from fastapi import FastAPI
from app.api.websocket import router as ws_router


def create_app():

    app = FastAPI(
        title="Driver Safety System",
        version="1.0.0"
    )

    # WebSocket routes
    app.include_router(ws_router)

    @app.get("/")
    async def health():
        return {
            "status": "Driver Safety Unified Server Running"
        }

    return app


app = create_app()