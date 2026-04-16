# Driver Drowsiness Detection (Debug-safe version)

import cv2
import pygame
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist
import os
import sys

print("Program started successfully")

# ── AUDIO SETUP ─────────────────────────────
try:
    pygame.mixer.init()
    print("Audio initialized")

    alert_path = "Assets/Alert.mp3"

    if os.path.exists(alert_path):
        pygame.mixer.music.load(alert_path)
        pygame.mixer.music.set_volume(0.3)
        print("Alert sound loaded")
    else:
        print("Alert.mp3 NOT found (continuing without sound)")

except Exception as e:
    print("Audio error:", e)


# ── MEDIAPIPE SETUP ─────────────────────────
try:
    mp_face_mesh = mp.solutions.face_mesh

    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    print("MediaPipe initialized")

except Exception as e:
    print("MediaPipe failed:", e)
    sys.exit()


LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

EAR_THRESHOLD = 0.22
CLOSED_FRAMES = 12


def compute_ear(landmarks, eye_indices, w, h):

    pts = []

    for idx in eye_indices:
        lm = landmarks[idx]
        pts.append((lm.x * w, lm.y * h))

    pts = np.array(pts)

    A = dist.euclidean(pts[1], pts[5])
    B = dist.euclidean(pts[2], pts[4])
    C = dist.euclidean(pts[0], pts[3])

    if C == 0:
        return 0

    return (A + B) / (2 * C)


# ── CAMERA SETUP ────────────────────────────
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Camera NOT detected")
    sys.exit()

print("Camera opened successfully")


# ── MAIN LOOP ───────────────────────────────
eye_closed_frames = 0
alarm_on = False

print("Entering detection loop")

while True:

    ret, frame = cap.read()

    if not ret:
        print("Frame capture failed")
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)

    eyes_open = False

    if results.multi_face_landmarks:

        landmarks = results.multi_face_landmarks[0].landmark

        left_ear = compute_ear(landmarks, LEFT_EYE, w, h)
        right_ear = compute_ear(landmarks, RIGHT_EYE, w, h)

        avg_ear = (left_ear + right_ear) / 2

        if avg_ear > EAR_THRESHOLD:
            eyes_open = True

        cv2.putText(frame,
                    f"EAR: {avg_ear:.2f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255,255,255),
                    2)

    if not eyes_open:
        eye_closed_frames += 1
    else:
        eye_closed_frames = 0

        if alarm_on:
            pygame.mixer.music.stop()
            alarm_on = False

    if eye_closed_frames >= CLOSED_FRAMES:

        if not alarm_on and os.path.exists("Assets/Alert.mp3"):
            pygame.mixer.music.play(-1)
            alarm_on = True


    status = "DROWSY!" if alarm_on else ("Eyes Open" if eyes_open else "Eyes Closed")

    cv2.putText(frame,
                status,
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0,0,255) if alarm_on else (0,255,0),
                2)

    cv2.imshow("Driver Monitor", frame)

    if cv2.waitKey(1) & 0xFF == ord('x'):
        break


cap.release()
face_mesh.close()
pygame.mixer.music.stop()
cv2.destroyAllWindows()