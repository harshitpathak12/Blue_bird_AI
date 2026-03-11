import numpy as np
import time

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

    def __init__(self):
        """
        Fatigue model based on EAR/MAR over a short confirmation window.
        Expects shared landmarks from FaceDetector — no duplicate detector.
        """
        self.fatigue_start_time = None
        self.fatigue_active = False
        self.last_ear = 0.0
        self.last_mar = 0.0
        print("ModelFatigue ready.")

    def process(self, frame, landmarks=None, img_w=None, img_h=None):
        current_time = time.time()
        fatigue_condition = False

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