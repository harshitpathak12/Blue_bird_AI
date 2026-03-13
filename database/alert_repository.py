from datetime import datetime, timezone
from database.mongodb_client import alerts_collection


def insert_alert(
    driver_id: str,
    alert_type: str,
    confidence_score: float,
    session_id: str | None = None,
    gps_latitude: float | None = None,
    gps_longitude: float | None = None,
):
    """
    Insert alert per PDF: driver_id, session_id, alert_type, confidence_score, timestamp, GPS.
    """
    alert = {
        "driver_id": driver_id,
        "session_id": session_id,
        "alert_type": alert_type,
        "confidence_score": confidence_score,
        "timestamp": datetime.now(timezone.utc),
    }
    if gps_latitude is not None and gps_longitude is not None:
        alert["gps"] = {"latitude": gps_latitude, "longitude": gps_longitude}
    try:
        result = alerts_collection.insert_one(alert)
        alert["_id"] = result.inserted_id
        return alert
    except Exception as e:
        print(f"[alert_repository] insert_one failed: {e}")
        raise


def get_alerts(driver_id: str | None = None, session_id: str | None = None, limit: int = 100):
    """List alerts, optionally by driver_id and/or session_id."""
    query = {}
    if driver_id:
        query["driver_id"] = driver_id
    if session_id:
        query["session_id"] = session_id
    cursor = alerts_collection.find(query).sort("timestamp", -1).limit(limit)
    return list(cursor)