"""
Match face embedding to driver in MongoDB.
Used by login (verify/identify) and by the real-time pipeline (recognize driver from frame).
"""

from typing import Optional, Tuple

import numpy as np

from database import driver_repository

# Minimum similarity to consider a driver as recognized (align with ArcFace THRESHOLD)
RECOGNITION_THRESHOLD = 0.45


def match_embedding_to_driver(
    embedding: np.ndarray,
    driver_id: Optional[str] = None,
) -> Tuple[Optional[dict], float]:
    """
    Match embedding against stored drivers in MongoDB.

    - If driver_id is provided: verify against that driver only; return (driver, score).
    - If driver_id is None: find best match across all drivers with face_embedding.

    Returns (driver document or None, similarity score in [0, 1]).
    """
    best_driver = None
    best_score = -1.0
    emb = np.asarray(embedding, dtype=np.float32)
    emb_norm = np.linalg.norm(emb) + 1e-10

    if driver_id:
        driver = driver_repository.get_driver_by_id(driver_id)
        if not driver or not driver.get("face_embedding"):
            return None, -1.0
        db_emb = np.array(driver["face_embedding"], dtype=np.float32)
        score = float(np.dot(emb, db_emb) / (np.linalg.norm(db_emb) + 1e-10))
        return (driver if score >= RECOGNITION_THRESHOLD else None), score

    for driver in driver_repository.get_all_drivers():
        stored = driver.get("face_embedding")
        if not stored:
            continue
        db_emb = np.array(stored, dtype=np.float32)
        score = float(np.dot(emb, db_emb) / (np.linalg.norm(db_emb) + 1e-10))
        if score > best_score:
            best_score = score
            best_driver = driver

    if best_driver is None or best_score < RECOGNITION_THRESHOLD:
        return None, best_score if best_driver is None else best_score
    return best_driver, best_score
