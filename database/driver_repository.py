import random
import string
from datetime import datetime, timezone

from database.mongodb_client import drivers_collection

_ID_CHARS = string.ascii_uppercase + string.digits  # A-Z 0-9
_ID_LENGTH = 9


def get_driver_by_id(driver_id: str):
    return drivers_collection.find_one({"driver_id": driver_id})


def get_all_drivers():
    """Return a cursor over all drivers."""
    return drivers_collection.find()


def _generate_unique_driver_id() -> str:
    """
    Generate a unique 9-character alphanumeric driver_id (A-Z, 0-9).

    Each ID is guaranteed unique across all registered drivers.
    Example: "A7K3M9X2B"
    """
    while True:
        candidate = "".join(random.choices(_ID_CHARS, k=_ID_LENGTH))
        if not drivers_collection.find_one({"driver_id": candidate}):
            return candidate


def create_driver(
    driver_id: str | None,
    name: str,
    age: int | None = None,
    face_embedding: list | None = None,
    face_embedding_3d: list | None = None,
    face_image_path: str | None = None,
):
    """
    Create driver record: driver_id, name, age, face_embedding (2D), face_embedding_3d, face_image_path, created_at.

    If driver_id is None, a unique 9-character alphanumeric ID is generated.
    """
    if driver_id is None:
        driver_id = _generate_unique_driver_id()

    driver = {
        "driver_id": driver_id,
        "name": name,
        "age": age,
        "face_embedding": face_embedding,
        "face_embedding_3d": face_embedding_3d,
        "face_image_path": face_image_path,
        "created_at": datetime.now(timezone.utc),
    }
    drivers_collection.insert_one(driver)
    return driver


def update_last_seen(driver_id: str):
    drivers_collection.update_one(
        {"driver_id": driver_id},
        {"$set": {"last_seen": datetime.now(timezone.utc)}},
    )