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

# ------------- Camera Setup -----------------

print("[INFO] Starting camera...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

timestamp = 0
confidence_value = 0.0

pred_queue: Deque[str] = deque(maxlen=PREDICTION_WINDOW)

# ------------- Main Loop -----------------

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
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

    # Exit hint
    panel_width, panel_height = 240, 50
    panel_x, panel_y = w - panel_width - 20, h - panel_height - 20
    draw_glass_panel(frame, panel_x, panel_y, panel_width, panel_height, 1)
    cv2.putText(frame, "Press ESC to Exit", (panel_x + 20, panel_y + 32), font, 0.6, (0, 255, 255), 2)

    cv2.imshow("Hand Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# ------------- Cleanup -----------------

cap.release()
cv2.destroyAllWindows()
print("[INFO] Finished.")