import cv2
import time
import subprocess
import shutil
from collections import deque
from pathlib import Path
import os
import streamlit as st
import mediapipe as mp
from src.inference_utils import extract_landmarks, get_stable_prediction, normalize_hand_xy_inference, draw_glass_panel, draw_progress_bar
from src.config import HAND_CONNECTIONS, VIDEO_CODEC, VIDEO_DEFAULT_FPS
from .model_utils import create_landmarker


def _reencode_to_h264(video_path: str) -> None:
    """Re-encode a video to H.264 (libx264) so browsers can play it.

    OpenCV's VideoWriter on Linux often falls back to mp4v which
    browsers cannot decode.  If ffmpeg is available we re-encode
    in-place; otherwise we silently skip (the user can still download).
    """
    if not os.path.exists(video_path):
        return
    if not shutil.which("ffmpeg"):
        return  # ffmpeg not installed – skip silently

    tmp_path = video_path + ".tmp.mp4"
    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", video_path,
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "23",
                "-movflags", "+faststart",
                "-an",              # no audio track needed
                tmp_path,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        os.replace(tmp_path, video_path)   # atomic swap
    except (subprocess.CalledProcessError, OSError):
        # If re-encoding fails, keep the original file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def annotate_frame(frame, model, encoder, landmarker, pred_queue, timestamp, conf_threshold, hud_scale=1.0, current_fps=0.0):
    # ...existing code from streamlit_app.py...
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect_for_video(mp_image, timestamp)

    label_text = "No Hand"
    confidence_value = 1.0

    if result.hand_landmarks:
        hand_lm = result.hand_landmarks[0]
        features = extract_landmarks(hand_lm, w, h)
        features_norm = normalize_hand_xy_inference(features)

        pred_encoded = model.predict(features_norm)[0]
        pred_label = str(encoder.inverse_transform([pred_encoded])[0])

        if hasattr(model, "predict_proba"):
            confidence_value = float(model.predict_proba(features_norm).max())
            if confidence_value < conf_threshold:
                pred_label = "Uncertain"

        pred_queue.append(pred_label)
        stable = get_stable_prediction(pred_queue)
        if stable:
            label_text = stable

        px_coords = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lm]
        for i, j in HAND_CONNECTIONS:
            cv2.line(frame, px_coords[i], px_coords[j], (0, 255, 255), 2)
        for pt in px_coords:
            cv2.circle(frame, pt, 4, (255, 0, 255), -1)

    # HUD overlay — equal padding on all sides
    panel_x, panel_y = 10, 10
    pad = max(int(14 * hud_scale), 8)
    panel_w = int(300 * hud_scale)
    text_x = panel_x + pad
    gesture_y = panel_y + pad + int(16 * hud_scale)
    conf_y = gesture_y + int(24 * hud_scale)
    bar_y = conf_y + int(16 * hud_scale)
    bar_w = panel_w - 2 * pad
    bar_h = max(int(12 * hud_scale), 7)
    panel_h = (bar_y + bar_h + pad) - panel_y

    font_scale_main = max(0.65 * hud_scale, 0.35)
    font_scale_sub = max(0.55 * hud_scale, 0.32)
    thickness = 2 if hud_scale >= 0.8 else 1

    draw_glass_panel(frame, panel_x, panel_y, panel_w, panel_h, 1)
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(frame, f"Gesture: {label_text}", (text_x, gesture_y), font, font_scale_main, (0, 255, 0), thickness)
    cv2.putText(frame, f"Confidence: {confidence_value * 100:.1f}%", (text_x, conf_y), font, font_scale_sub, (0, 255, 0), thickness)
    draw_progress_bar(frame, text_x, bar_y, bar_w, bar_h, confidence_value)

    return frame, label_text, confidence_value

def _process_video(input_path, output_path, model, encoder, conf_threshold, pred_window):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        st.error("Could not open the uploaded video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or VIDEO_DEFAULT_FPS
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    out = None
    for codec in ("avc1", "mp4v", VIDEO_CODEC):
        fourcc = cv2.VideoWriter_fourcc(*codec)
        candidate = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if candidate.isOpened():
            out = candidate
            break
        candidate.release()

    if out is None:
        st.error("Could not initialize video writer for output.")
        cap.release()
        return

    pred_queue: deque = deque(maxlen=pred_window)
    landmarker = create_landmarker()
    timestamp = 0
    processed = 0
    started = time.time()

    progress_bar = st.progress(0)
    status_text = st.empty()
    preview_slot = st.empty()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp += 1
            frame, _, _ = annotate_frame(
                frame, model, encoder, landmarker, pred_queue, timestamp, conf_threshold, hud_scale=0.55, current_fps=0.0
            )
            out.write(frame)
            processed += 1

            pct = min(processed / total_frames, 1.0)
            progress_bar.progress(pct)
            status_text.caption(f"Annotating frame {processed} of {total_frames}...")

            if processed % 15 == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                preview_slot.image(rgb, channels="RGB", use_container_width=True)
    finally:
        cap.release()
        out.release()
        landmarker.close()

    elapsed = max(time.time() - started, 1e-6)
    progress_bar.empty()
    status_text.empty()

    # Re-encode to H.264 for browser compatibility (mp4v is not playable in browsers)
    _reencode_to_h264(output_path)

    # Display the processed video after completion
    if os.path.exists(output_path):
        with open(output_path, "rb") as f:
            preview_slot.video(f)