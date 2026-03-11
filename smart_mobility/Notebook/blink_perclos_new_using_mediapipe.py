import cv2
import numpy as np
import time
from collections import deque
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ---------------- CONFIG ----------------
MODEL_PATH = "face_landmarker.task"
EAR_THRESHOLD = 0.22
CONSEC_FRAMES = 2
PERCLOS_WINDOW = 30 
DROWSY_THRESHOLD = 0.4

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# ------------ EAR FUNCTION --------------
def compute_ear(eye):
    p1, p2, p3, p4, p5, p6 = eye
    vertical1 = np.linalg.norm(p2 - p6)
    vertical2 = np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4)
    if horizontal == 0:
        return 0
    return (vertical1 + vertical2) / (2.0 * horizontal)

# ------------ Setup Face Landmarker --------------
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1
)

detector = vision.FaceLandmarker.create_from_options(options)

# ------------ Main --------------
def main():
    cap = cv2.VideoCapture(0)

    blink_count = 0
    frame_counter = 0
    closed_frames = deque()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        result = detector.detect(mp_image)

        ear = 0
        eye_closed = False

        if result.face_landmarks:
            landmarks = result.face_landmarks[0]

            left_eye = np.array(
                [[int(landmarks[i].x * w),
                  int(landmarks[i].y * h)] for i in LEFT_EYE]
            )

            right_eye = np.array(
                [[int(landmarks[i].x * w),
                  int(landmarks[i].y * h)] for i in RIGHT_EYE]
            )

            left_ear = compute_ear(left_eye)
            right_ear = compute_ear(right_eye)
            ear = (left_ear + right_ear) / 2

            cv2.polylines(frame, [left_eye], True, (0,255,0), 1)
            cv2.polylines(frame, [right_eye], True, (0,255,0), 1)

            if ear < EAR_THRESHOLD:
                frame_counter += 1
                eye_closed = True
            else:
                if frame_counter >= CONSEC_FRAMES:
                    blink_count += 1
                frame_counter = 0

        # -------- PERCLOS --------
        now = time.time()
        closed_frames.append((now, eye_closed))

        while closed_frames and now - closed_frames[0][0] > PERCLOS_WINDOW:
            closed_frames.popleft()

        total = len(closed_frames)
        closed = sum(1 for t, c in closed_frames if c)
        perclos = closed / total if total > 0 else 0

        # -------- DISPLAY --------
        cv2.putText(frame, f"EAR: {ear:.2f}", (20,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        cv2.putText(frame, f"Blinks: {blink_count}", (20,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        cv2.putText(frame, f"PERCLOS: {perclos:.2f}", (20,90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        if perclos > DROWSY_THRESHOLD:
            cv2.putText(frame, "DROWSINESS ALERT!",
                        (100,150),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0,0,255),
                        3)

        cv2.imshow("Blink & PERCLOS - New API", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
