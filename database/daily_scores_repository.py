"""
Daily safety scores per PDF: driver_id, date, fatigue_count, distraction_count, sleep_count, safety_score.
"""

from datetime import date, datetime, timezone
from database.mongodb_client import daily_scores_collection


def upsert_daily_score(
    driver_id: str,
    score_date: date,
    fatigue_count: int = 0,
    distraction_count: int = 0,
    sleep_count: int = 0,
    safety_score: float = 0.0,
):
    """Insert or replace daily score for a driver on a given date."""
    doc = {
        "driver_id": driver_id,
        "date": score_date.isoformat(),
        "fatigue_count": fatigue_count,
        "distraction_count": distraction_count,
        "sleep_count": sleep_count,
        "safety_score": safety_score,
        "updated_at": datetime.now(timezone.utc),
    }
    daily_scores_collection.update_one(
        {"driver_id": driver_id, "date": score_date.isoformat()},
        {"$set": doc},
        upsert=True,
    )
    return doc


def get_daily_scores(driver_id: str, date_from: str | None = None, date_to: str | None = None):
    """Get daily scores for a driver, optionally filtered by date range (YYYY-MM-DD)."""
    query = {"driver_id": driver_id}
    if date_from and date_to:
        query["date"] = {"$gte": date_from, "$lte": date_to}
    elif date_from:
        query["date"] = {"$gte": date_from}
    elif date_to:
        query["date"] = {"$lte": date_to}
    cursor = daily_scores_collection.find(query).sort("date", -1)
    return list(cursor)
