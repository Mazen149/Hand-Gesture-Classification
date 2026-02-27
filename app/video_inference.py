import os
import sys
import cv2
import numpy as np
import joblib
import mediapipe as mp

from collections import deque
from typing import Deque

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.inference_utils import (
    normalize_hand_xy_inference,
    extract_landmarks,
    get_stable_prediction,
    draw_glass_panel,
    draw_progress_bar
)

# ------------- Config -----------------

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

MODEL_PATH = os.path.join(BASE_DIR, "models", "HandGesture_XGBoost_Shallow.joblib")
ENCODER_PATH = os.path.join(BASE_DIR, "models", "label_encoder.joblib")
TASK_MODEL_PATH = os.path.join(BASE_DIR, "models", "hand_landmarker.task")

PREDICTION_WINDOW = 15
CONFIDENCE_THRESHOLD = 0.5

# ------------- Ask User For Video Path -----------------

INPUT_VIDEO_PATH = input("\n📁 Enter full path of the input video:\n> ").strip()

if not os.path.exists(INPUT_VIDEO_PATH):
    raise FileNotFoundError("The provided video path does not exist.")

video_name = os.path.splitext(os.path.basename(INPUT_VIDEO_PATH))[0]
OUTPUT_VIDEO_PATH = os.path.join(BASE_DIR, f"{video_name}_prediction.mp4")

# ------------- Load Model + Encoder -----------------

print("[INFO] Loading model...")
model = joblib.load(MODEL_PATH)

print("[INFO] Loading encoder...")
encoder = joblib.load(ENCODER_PATH)

# ------------- MediaPipe Setup -----------------

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=TASK_MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
)

landmarker = HandLandmarker.create_from_options(options)

CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
]

# ------------- Video Setup -----------------

print("[INFO] Opening video...")
cap = cv2.VideoCapture(INPUT_VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError("Could not open input video")

TARGET_WIDTH = 1920
TARGET_HEIGHT = 1080

fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 30

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (TARGET_WIDTH, TARGET_HEIGHT))

timestamp = 0
confidence_value = 0.0
pred_queue: Deque[str] = deque(maxlen=PREDICTION_WINDOW)

print("[INFO] Processing video...")

# ------------- Main Loop -----------------

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize to Full HD
    frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))
    h, w, _ = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    timestamp += 1
    results = landmarker.detect_for_video(mp_image, timestamp)

    label_text = "No Hand"
    confidence_value = 1.0

    if results.hand_landmarks:
        hand_landmarks = results.hand_landmarks[0]

        features = extract_landmarks(hand_landmarks)
        features_norm = normalize_hand_xy_inference(features)

        pred_encoded = model.predict(features_norm)[0]
        pred_label = encoder.inverse_transform([pred_encoded])[0]

        if hasattr(model, "predict_proba"):
            confidence_value = float(model.predict_proba(features_norm).max())
            if confidence_value < CONFIDENCE_THRESHOLD:
                pred_label = "Uncertain"

        pred_queue.append(pred_label)
        stable_pred = get_stable_prediction(pred_queue)

        if stable_pred:
            label_text = stable_pred

        pixel_coords = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]

        for connection in CONNECTIONS:
            cv2.line(frame, pixel_coords[connection[0]], pixel_coords[connection[1]], (0,255,255), 2)
        for coord in pixel_coords:
            cv2.circle(frame, coord, 4, (255,0,255), -1)

    # ------------- HUD -----------------

    draw_glass_panel(frame, 10, 10, 320, 140, 1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f"Gesture: {label_text}", (22, 50), font, 0.75, (0, 255, 0), 2)
    cv2.putText(frame, f"Confidence: {confidence_value*100:.1f}%", (22, 85), font, 0.65, (0, 255, 0), 2)
    draw_progress_bar(frame, 22, 105, 260, 14, confidence_value)

    # ------------- Write & Show -----------------

    out.write(frame)
    cv2.imshow("Hand Gesture Recognition - Video Mode", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# ------------- Cleanup -----------------

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"[INFO] Video saved to: {OUTPUT_VIDEO_PATH}")