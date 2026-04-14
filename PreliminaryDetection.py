# Driver Drowsiness Detection using Haar Cascades + Alarm Sound (Final Version)

import cv2
import pygame

pygame.mixer.init()
pygame.mixer.music.load("Assets/Alert.mp3")
pygame.mixer.music.set_volume(0.1)

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml"
)

eye_closed_frames = 0
threshold = 5
alarm_on = False

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    eyes_detected = False

    for (x, y, w, h) in faces:

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        roi_gray = gray[y:y + int(h * 0.6), x:x+w]
        roi_color = frame[y:y + int(h * 0.6), x:x+w]

        eyes = eye_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.15,
            minNeighbors=6,
            minSize=(18, 18)
        )

        if len(eyes) > 0:
            eyes_detected = True

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(
                roi_color,
                (ex, ey),
                (ex+ew, ey+eh),
                (0, 255, 0),
                2
            )

    if not eyes_detected:
        eye_closed_frames += 1
    else:
        eye_closed_frames = 0

        if alarm_on:
            pygame.mixer.music.stop()
            alarm_on = False

    if eye_closed_frames >= threshold and not alarm_on:
        pygame.mixer.music.play(-1)
        alarm_on = True

    cv2.imshow("Driver Monitor", frame)

    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

cap.release()
pygame.mixer.music.stop()
cv2.destroyAllWindows()