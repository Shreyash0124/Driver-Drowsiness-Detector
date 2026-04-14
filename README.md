# Real-Time Driver Drowsiness Detection and Risk Alert System Using Computer Vision

## Project Overview

Driver drowsiness is one of the leading causes of road accidents globally. Existing systems detect
fatigue using a single signal — lane deviation or blink rate — leading to delayed and inaccurate alerts.

This project proposes a real-time multi-cue drowsiness detection system that uses computer vision and
temporal AI to fuse eye closure (EAR), yawning (MAR), and head tilt data, and immediately trigger
tiered alerts to the driver and authorities.

---

## Problem Statement

Traditional drowsiness detection systems rely on a single data source and produce reactive,
threshold-based alerts only after the risk has already occurred. A real-time system that continuously
analyzes multiple facial behavioral cues using intelligent temporal pattern recognition is needed for
early and reliable driver fatigue warnings.

---

## Objectives

- Develop a real-time facial monitoring system using a camera interfaced with IoT hardware (Raspberry Pi)
- Extract multi-cue drowsiness indicators: Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR), head pose angle
- Apply a CNN + LSTM model to detect temporal drowsiness patterns across frames
- Generate immediate tiered alerts: audio buzzer, LED indicator, and SMS/app notification

---

---

## AI Technique Used

| Technique | Purpose |
|---|---|
| MediaPipe Face Mesh / Dlib | 68-point facial landmark detection |
| Eye Aspect Ratio (EAR) | Detect prolonged eye closure (blink frequency) |
| Mouth Aspect Ratio (MAR) | Detect yawning events |
| Head Pose Estimation | Detect nodding or drooping head posture |
| CNN | Spatial feature extraction from facial frames |
| LSTM | Temporal pattern detection across frame sequences |

---

## Innovation / Novelty

1. **Multi-cue fusion** — combines EAR, MAR, and head pose instead of relying on a single signal
2. **Temporal AI (LSTM)** — detects gradual drowsiness progression over time, not just single-frame thresholds
3. **Real-time edge deployment** — runs on Raspberry Pi for in-vehicle use without cloud dependency
4. **Tiered alert system** — escalates from visual → audio → remote SMS based on severity level

---

## Limitations of Existing Systems

| Existing System | Limitation |
|---|---|
| Steering pattern monitors | Detects only after vehicle behavior changes |
| Lane deviation warnings | Reactive, not predictive |
| Single eye-blink detectors | Fails in low light or glasses use |
| Wearable EEG monitors | Not practical for everyday drivers |

---

## Tech Stack

- **Hardware:** Raspberry Pi 4 / Laptop webcam, USB camera, buzzer, LED
- **Languages:** Python 3.x
- **Libraries:** OpenCV, MediaPipe, Dlib, NumPy, TensorFlow / Keras, scikit-learn
- **Alert Integration:** Twilio API (SMS), GPIO (LED + buzzer)
- **Dataset:** NTHU Drowsy Driver Detection Dataset / custom captured data

---

## Features

- Real-time 30fps facial landmark tracking
- EAR threshold-based blink detection with configurable sensitivity
- MAR threshold-based yawn detection
- Head pose angle estimation using solvePnP
- LSTM-based temporal classifier trained on frame sequences
- Three alert tiers: Safe (green LED), Warning (yellow + beep), Alert (red + buzzer + SMS)

---

## Project Status

| Phase | Status |
|---|---|
| Literature review | ✅ Completed |
| System architecture design | ✅ Completed |
| EAR / MAR detection module | 🔄 In Progress |
| LSTM model training | ⬜ Pending |
| Hardware integration | ⬜ Pending |
| Alert system integration | ⬜ Pending |
| Testing and validation | ⬜ Pending |

---

## Demo (In Progress)

Current demo includes:
- Live webcam feed with facial landmark overlay
- Real-time EAR and MAR value display
- Basic threshold-based drowsiness detection with audio alert

---

## How to Run (Demo)

```bash
git clone https://github.com/yourusername/drowsiness-detection
cd drowsiness-detection
pip install -r requirements.txt
python detect_drowsiness.py
```

---

## Future Scope

- Integration with vehicle OBD-II port for speed-aware alert calibration
- Cloud dashboard for fleet-level fatigue monitoring
- Transfer learning with pre-trained models (MobileNet) for improved accuracy
- Night-vision camera support for low-light environments

---

## References

1. Soukupova, T. & Cech, J. (2016). Real-Time Eye Blink Detection using Facial Landmarks
2. NTHU Drowsy Driver Detection Dataset
3. MediaPipe Face Mesh Documentation — Google AI
4. OpenCV Documentation — opencv.org