"""
Session start/end – used after login.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.schemas.payloads import SessionEndResponse, SessionStartResponse
from database import session_repository

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


class SessionStartRequest(BaseModel):
    driver_id: str


class SessionEndRequest(BaseModel):
    session_id: str


@router.post("/start", response_model=SessionStartResponse)
async def start_session(body: SessionStartRequest):
    session_id = session_repository.create_session(body.driver_id)
    session = session_repository.get_session(session_id)
    return SessionStartResponse(
        session_id=session_id,
        driver_id=body.driver_id,
        start_time=session["start_time"],
        status="active",
        message="Session started",
    )


@router.post("/end", response_model=SessionEndResponse)
async def end_session(body: SessionEndRequest):
    session = session_repository.get_session(body.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    from datetime import datetime
    end_time = datetime.utcnow()
    session_repository.end_session(body.session_id)
    return SessionEndResponse(
        session_id=body.session_id,
        end_time=end_time,
        status="ended",
        message="Session ended",
    )
