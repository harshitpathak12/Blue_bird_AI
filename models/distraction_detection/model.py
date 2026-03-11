import os

import cv2
import numpy as np
import joblib
import time


ROOT_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
BASE_MODEL_DIR = os.getenv(
    "MODEL_BASE_DIR",
    os.path.join(ROOT_DIR, "fine_tunned_pre_train"),
)


class ModelHeadPose:

    def __init__(self, classifier_path: str | None = None):

        if classifier_path is None:
            classifier_path = os.path.join(BASE_MODEL_DIR, "headpose_classifier.pkl")

        try:
            self.classifier = joblib.load(classifier_path)
        except FileNotFoundError:
            print(
                f"[ModelHeadPose] classifier file not found at '{classifier_path}'. "
                "Head pose classification will be disabled until the file is provided."
            )
            self.classifier = None

        self.last_prediction = "Forward"
        self.last_pitch = self.last_yaw = self.last_roll = 0.0

        # Track how long the head has been turned away from the road
        self.head_turned_away_sec = 0.0
        self._away_start_time = None

    def process(self, frame, landmarks, img_w, img_h):

        if self.classifier is None:
            # No classifier available; return frame unchanged and keep defaults.
            return frame
        face_2d = []
        face_3d = []
        landmark_ids = [33, 263, 1, 61, 291, 199]

        for idx in landmark_ids:
            lm = landmarks[idx]
            x, y = int(lm.x * img_w), int(lm.y * img_h)
            face_2d.append([x, y])
            face_3d.append([x, y, lm.z])

        face_2d = np.array(face_2d, dtype=np.float64)
        face_3d = np.array(face_3d, dtype=np.float64)

        focal_length = img_w
        cam_matrix = np.array(
            [
                [focal_length, 0, img_w / 2],
                [0, focal_length, img_h / 2],
                [0, 0, 1],
            ]
        )

        dist_matrix = np.zeros((4, 1))

        success, rot_vec, _ = cv2.solvePnP(
            face_3d,
            face_2d,
            cam_matrix,
            dist_matrix,
        )

        if not success:
            self.last_pitch = self.last_yaw = self.last_roll = 0.0
            return frame

        rmat, _ = cv2.Rodrigues(rot_vec)
        angles, *_ = cv2.RQDecomp3x3(rmat)

        pitch = angles[0] * 360
        yaw = angles[1] * 360
        roll = angles[2] * 360

        prediction = self.classifier.predict([[pitch, yaw, roll]])[0]

        self.last_prediction = prediction
        self.last_pitch = pitch
        self.last_yaw = yaw
        self.last_roll = roll

        # Update away duration timer
        now = time.time()
        if prediction.lower() == "forward":
            self._away_start_time = None
            self.head_turned_away_sec = 0.0
        else:
            if self._away_start_time is None:
                self._away_start_time = now
                self.head_turned_away_sec = 0.0
            else:
                self.head_turned_away_sec = now - self._away_start_time

        # HUD drawn by app.utils.overlay
        return frame