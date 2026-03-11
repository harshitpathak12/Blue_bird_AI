import os

import numpy as np
import joblib
from collections import deque


ROOT_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
BASE_MODEL_DIR = os.getenv(
    "MODEL_BASE_DIR",
    os.path.join(ROOT_DIR, "fine_tunned_pre_train"),
)


class ModelEyeGaze:

    FEATURE_COLUMNS = [
        "r_h",
        "r_v",
        "l_h",
        "l_v",
        "avg_h",
        "avg_v",
        "diff_h",
        "diff_v",
    ]

    RIGHT_EYE_IDX = {
        "LEFT": 33,
        "RIGHT": 133,
        "TOP": 159,
        "BOTTOM": 145,
        "IRIS": 468,
    }

    LEFT_EYE_IDX = {
        "LEFT": 362,
        "RIGHT": 263,
        "TOP": 386,
        "BOTTOM": 374,
        "IRIS": 473,
    }

    def __init__(
        self,
        classifier_path: str | None = None,
        scaler_path: str | None = None,
    ):

        if classifier_path is None:
            classifier_path = os.path.join(BASE_MODEL_DIR, "eye_gaze.pkl")
        if scaler_path is None:
            scaler_path = os.path.join(BASE_MODEL_DIR, "eye_scaler.pkl")

        try:
            self.classifier = joblib.load(classifier_path)
            self.scaler = joblib.load(scaler_path)
        except FileNotFoundError:
            print(
                f"[ModelEyeGaze] classifier or scaler file not found "
                f"('{classifier_path}', '{scaler_path}'). "
                "Eye gaze classification will be disabled until the files are provided."
            )
            self.classifier = None
            self.scaler = None

        # Feature smoothing (EMA)
        self.alpha = 0.6
        self.prev_features = None

        # Prediction stabilization
        self.pred_buffer = deque(maxlen=10)
        self.stable_prediction = "CENTER"

    # ------------------------------------------------
    # Utility Functions
    # ------------------------------------------------

    def _extract_eye_features(self, landmarks, img_w, img_h, eye_idx):
        lm = landmarks
        left_x = lm[eye_idx["LEFT"]].x * img_w
        left_y = lm[eye_idx["LEFT"]].y * img_h
        right_x = lm[eye_idx["RIGHT"]].x * img_w
        right_y = lm[eye_idx["RIGHT"]].y * img_h
        top_x = lm[eye_idx["TOP"]].x * img_w
        top_y = lm[eye_idx["TOP"]].y * img_h
        bottom_x = lm[eye_idx["BOTTOM"]].x * img_w
        bottom_y = lm[eye_idx["BOTTOM"]].y * img_h
        iris_x = lm[eye_idx["IRIS"]].x * img_w
        iris_y = lm[eye_idx["IRIS"]].y * img_h

        width = np.hypot(right_x - left_x, right_y - left_y)
        height = np.hypot(bottom_x - top_x, bottom_y - top_y)

        if width == 0 or height == 0:
            return None

        h_ratio = np.clip((iris_x - left_x) / width, 0, 1)
        v_ratio = np.clip((iris_y - top_y) / height, 0, 1)

        return h_ratio, v_ratio

    def _smooth(self, current):
        if self.prev_features is None:
            self.prev_features = current
            return current

        smoothed = self.alpha * current + (1 - self.alpha) * self.prev_features
        self.prev_features = smoothed
        return smoothed

    def _stabilize_prediction(self, pred):
        self.pred_buffer.append(pred)
        self.stable_prediction = max(set(self.pred_buffer),
                                     key=self.pred_buffer.count)
        return self.stable_prediction

    # ------------------------------------------------
    # Main Processing
    # ------------------------------------------------

    def process(self, frame, landmarks, img_w, img_h):

        if self.classifier is None or self.scaler is None:
            return frame

        right = self._extract_eye_features(landmarks, img_w, img_h, self.RIGHT_EYE_IDX)
        left = self._extract_eye_features(landmarks, img_w, img_h, self.LEFT_EYE_IDX)

        if right is None or left is None:
            return frame

        r_h, r_v = right
        l_h, l_v = left

        avg_h = (r_h + l_h) / 2
        avg_v = (r_v + l_v) / 2

        feature_array = np.array([
            r_h, r_v, l_h, l_v,
            avg_h, avg_v, r_h - l_h, r_v - l_v,
        ])

        feature_array = self._smooth(feature_array)

        scaled = self.scaler.transform(feature_array.reshape(1, -1))
        raw_prediction = self.classifier.predict(scaled)[0]
        self._stabilize_prediction(raw_prediction)

        return frame