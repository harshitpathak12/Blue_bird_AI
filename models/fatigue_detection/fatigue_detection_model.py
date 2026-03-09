# models/model_fatigue.py

import os

import cv2
import numpy as np
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


ROOT_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
BASE_MODEL_DIR = os.getenv(
    "MODEL_BASE_DIR",
    os.path.join(ROOT_DIR, "fine_tunned_pre_train"),
)

# ---------------- CONFIG ---------------- #
EAR_THRESHOLD = 0.22
MAR_THRESHOLD = 0.45
FATIGUE_DURATION = 2.0  # seconds continuous

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Mouth landmarks
TOP_LIP = 13
BOTTOM_LIP = 14
LEFT_MOUTH = 78
RIGHT_MOUTH = 308
# ---------------------------------------- #


def compute_ear(eye):
    p1, p2, p3, p4, p5, p6 = eye
    vertical1 = np.linalg.norm(p2 - p6)
    vertical2 = np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4)
    if horizontal == 0:
        return 0.0
    return (vertical1 + vertical2) / (2.0 * horizontal)


def compute_mar(landmarks, w, h):
    top = np.array([landmarks[TOP_LIP].x * w, landmarks[TOP_LIP].y * h])
    bottom = np.array([landmarks[BOTTOM_LIP].x * w, landmarks[BOTTOM_LIP].y * h])
    left = np.array([landmarks[LEFT_MOUTH].x * w, landmarks[LEFT_MOUTH].y * h])
    right = np.array([landmarks[RIGHT_MOUTH].x * w, landmarks[RIGHT_MOUTH].y * h])

    vertical = np.linalg.norm(top - bottom)
    horizontal = np.linalg.norm(left - right)
    if horizontal == 0:
        return 0.0
    return vertical / horizontal


class ModelFatigue:

    def __init__(self, model_path: str | None = None):
        """
        Fatigue model based on EAR/MAR over a short confirmation window.

        By default it owns its own MediaPipe FaceLandmarker instance, but it can
        also reuse externally computed landmarks (e.g. from a shared FaceDetector)
        when they are passed into `process`.
        """
        print("Loading MediaPipe FaceLandmarker for fatigue model...")

        if model_path is None:
            # Allow overriding via env var, defaulting to fine_tunned_pre_train.
            model_path = os.getenv(
                "FACE_LANDMARKER_PATH",
                os.path.join(BASE_MODEL_DIR, "face_landmarker.task"),
            )

        try:
            base_options = python.BaseOptions(model_asset_path=model_path)

            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
                num_faces=1,
            )

            self.detector = vision.FaceLandmarker.create_from_options(options)
        except FileNotFoundError:
            print(
                f"[ModelFatigue] face_landmarker.task not found at '{model_path}'. "
                "Fatigue landmarks will be disabled until the model file is provided."
            )
            self.detector = None

        self.fatigue_start_time = None
        self.fatigue_active = False
        self.last_ear = 0.0
        self.last_mar = 0.0

        print("ModelFatigue ready.")

    def process(self, frame, landmarks=None, img_w=None, img_h=None):
        """
        Process a frame and update internal fatigue state.

        If `landmarks` (and image size) are provided, they are reused directly.
        Otherwise this method runs its own FaceLandmarker detection.
        """
        current_time = time.time()
        fatigue_condition = False

        # Use external landmarks when available to avoid duplicate detection.
        if landmarks is None:
            if self.detector is None:
                return frame
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self.detector.detect(mp_image)
            if result.face_landmarks:
                landmarks = result.face_landmarks[0]

        if landmarks is not None:
            h, w, _ = frame.shape

            # --------- EAR ---------
            left_eye = np.array(
                [[landmarks[i].x * w, landmarks[i].y * h] for i in LEFT_EYE]
            )
            right_eye = np.array(
                [[landmarks[i].x * w, landmarks[i].y * h] for i in RIGHT_EYE]
            )

            ear = (compute_ear(left_eye) + compute_ear(right_eye)) / 2.0
            self.last_ear = ear
            self.last_mar = compute_mar(landmarks, w, h)
            mar = self.last_mar

            # Fatigue condition
            if ear < EAR_THRESHOLD or mar > MAR_THRESHOLD:
                fatigue_condition = True

        # --------- 2 SECOND CONFIRMATION ---------
        if fatigue_condition:
            if self.fatigue_start_time is None:
                self.fatigue_start_time = current_time
            elif current_time - self.fatigue_start_time >= FATIGUE_DURATION:
                self.fatigue_active = True
        else:
            self.fatigue_start_time = None
            self.fatigue_active = False
            if landmarks is None:
                self.last_ear = 0.0
                self.last_mar = 0.0

        # HUD is drawn by app.utils.overlay
        return frame