import os

import cv2
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


class FaceDetector:

    def __init__(self, model_path: str | None = None):
        """
        Lightweight wrapper around MediaPipe FaceLandmarker.

        The model path can be provided explicitly or via the FACE_LANDMARKER_PATH
        environment variable. If the file is missing, the detector is disabled
        and get_landmarks will always return (None, None, None) instead of
        crashing the server.
        """
        if model_path is None:
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
                f"[FaceDetector] face_landmarker.task not found at '{model_path}'. "
                "Face landmarks will be disabled until the model file is provided."
            )
            self.detector = None

    def get_landmarks(self, frame):

        if self.detector is None:
            return None, None, None

        img_h, img_w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        result = self.detector.detect(mp_image)

        if not result.face_landmarks:
            return None, None, None

        return result.face_landmarks[0], img_w, img_h