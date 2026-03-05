import os
import sys
import cv2
import numpy as np
import joblib
import mediapipe as mp

from collections import deque
from typing import Deque

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".." , "..")))

from src.inference_utils import (
    normalize_hand_xy_inference,
    extract_landmarks,
    get_stable_prediction,
    draw_glass_panel,
    draw_progress_bar
)
from src.config import (
    MODEL_PATH,
    ENCODER_PATH,
    TASK_MODEL_PATH,
    PREDICTION_WINDOW,
    CONFIDENCE_THRESHOLD,
    HAND_CONNECTIONS,
    VIDEO_TARGET_WIDTH,
    VIDEO_TARGET_HEIGHT,
    VIDEO_DEFAULT_FPS,
    VIDEO_CODEC,
    BASE_DIR,
)

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

# ------------- Video Setup -----------------

print("[INFO] Opening video...")
cap = cv2.VideoCapture(INPUT_VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError("Could not open input video")

fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = VIDEO_DEFAULT_FPS

fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (VIDEO_TARGET_WIDTH, VIDEO_TARGET_HEIGHT))

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
    frame = cv2.resize(frame, (VIDEO_TARGET_WIDTH, VIDEO_TARGET_HEIGHT))
    h, w, _ = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    timestamp += 1
    results = landmarker.detect_for_video(mp_image, timestamp)

    label_text = "No Hand"
    confidence_value = 1.0

    if results.hand_landmarks:
        hand_landmarks = results.hand_landmarks[0]

        features = extract_landmarks(hand_landmarks, w, h)
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

        for connection in HAND_CONNECTIONS:
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