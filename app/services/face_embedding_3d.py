"""
Build a 3D face embedding from MediaPipe face landmarks for driver identification.
Pose/scale normalized, L2-normalized vector for cosine similarity matching.
"""

from typing import Any

import numpy as np

# MediaPipe Face Landmarker typically returns 478 landmarks; we use up to 468 for compatibility
_MIN_LANDMARKS = 10

# Indices for normalization (MediaPipe face mesh)
_LEFT_EYE_CENTER_IDS = [33, 133, 159, 145, 153, 144]
_RIGHT_EYE_CENTER_IDS = [362, 263, 386, 374, 373, 380]
_NOSE_TIP_IDX = 1


def _get_point(lm: Any) -> np.ndarray:
    return np.array([lm.x, lm.y, lm.z], dtype=np.float64)


def build_3d_embedding(landmarks: Any) -> np.ndarray | None:
    """
    Build a pose/scale-normalized 3D face embedding from MediaPipe landmarks.

    - Centers by centroid, scales by inter-ocular distance, L2-normalizes.
    - Returns a 1D float32 vector (length N*3, e.g. 1404 for 468 landmarks).
    - Returns None if landmarks is None or has too few points.
    """
    if landmarks is None:
        return None
    n = len(landmarks)
    if n < _MIN_LANDMARKS:
        return None

    # Flatten to (N, 3)
    pts = np.array([_get_point(landmarks[i]) for i in range(n)], dtype=np.float64)

    # Center
    centroid = np.mean(pts, axis=0)
    pts = pts - centroid

    # Scale by inter-ocular distance (or fallback to norm of first point)
    if n > max(max(_RIGHT_EYE_CENTER_IDS), max(_LEFT_EYE_CENTER_IDS)):
        left_center = np.mean(
            [pts[i] for i in _LEFT_EYE_CENTER_IDS if i < n],
            axis=0,
        )
        right_center = np.mean(
            [pts[i] for i in _RIGHT_EYE_CENTER_IDS if i < n],
            axis=0,
        )
        scale = np.linalg.norm(left_center - right_center)
    else:
        scale = np.linalg.norm(pts[_NOSE_TIP_IDX]) if _NOSE_TIP_IDX < n else 1.0

    if scale < 1e-8:
        scale = 1.0
    pts = pts / scale

    # Flatten to 1D and L2-normalize (for cosine similarity)
    vec = pts.ravel().astype(np.float32)
    norm = np.linalg.norm(vec) + 1e-10
    vec = vec / norm
    return vec
