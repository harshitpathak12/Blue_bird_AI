from datetime import datetime, timezone
from bson import ObjectId
from database.mongodb_client import sessions_collection


def create_session(driver_id: str):
    session = {
        "driver_id": driver_id,
        "start_time": datetime.now(timezone.utc),
        "end_time": None,
        "status": "active",
    }
    try:
        result = sessions_collection.insert_one(session)
        return str(result.inserted_id)
    except Exception as e:
        print(f"[session_repository] insert_one failed: {e}")
        raise


def get_session(session_id: str):
    """Get session by id (accepts string ObjectId)."""
    try:
        return sessions_collection.find_one({"_id": ObjectId(session_id)})
    except Exception:
        return None


def end_session(session_id: str):
    try:
        sessions_collection.update_one(
            {"_id": ObjectId(session_id)},
            {"$set": {"end_time": datetime.now(timezone.utc), "status": "ended"}},
        )
    except Exception:
        pass