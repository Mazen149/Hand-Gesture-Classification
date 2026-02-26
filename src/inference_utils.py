from typing import Deque, Optional, Sequence
import numpy as np
import pandas as pd
from collections import Counter
import cv2


# ------------- Normalization -----------------

def normalize_hand_xy_inference(
    flat_features: np.ndarray,
    wrist_idx: int = 1,
    mid_idx: int = 13,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Normalize flattened hand landmarks to be translation-
    and scale-invariant (same logic used in training).
    """

    cols = []
    for i in range(1, 22):
        cols.extend([f"x{i}", f"y{i}", f"z{i}"])

    X_df = pd.DataFrame([flat_features], columns=cols)

    x_cols = [f"x{i}" for i in range(1, 22)]
    y_cols = [f"y{i}" for i in range(1, 22)]

    wrist_x = X_df[f"x{wrist_idx}"].to_numpy()
    wrist_y = X_df[f"y{wrist_idx}"].to_numpy()

    X_df.loc[:, x_cols] = X_df.loc[:, x_cols].sub(wrist_x, axis=0)
    X_df.loc[:, y_cols] = X_df.loc[:, y_cols].sub(wrist_y, axis=0)

    mid_x = X_df[f"x{mid_idx}"].to_numpy()
    mid_y = X_df[f"y{mid_idx}"].to_numpy()

    scale = np.sqrt(mid_x**2 + mid_y**2)
    scale = np.maximum(scale, eps)

    X_df.loc[:, x_cols] = X_df.loc[:, x_cols].div(scale, axis=0)
    X_df.loc[:, y_cols] = X_df.loc[:, y_cols].div(scale, axis=0)

    return X_df.values


# ------------- Landmark Extraction -----------------

def extract_landmarks(
    hand_landmarks: Sequence,
) -> np.ndarray:
    """
    Flatten 21 MediaPipe landmarks into a 63-dim vector.
    """
    coords: list[float] = []

    for lm in hand_landmarks:
        coords.extend([lm.x, lm.y, lm.z])

    return np.array(coords, dtype=np.float32)


# ------------- Prediction Stability -----------------

def get_stable_prediction(
    queue: Deque[str],
) -> Optional[str]:
    """
    Return the most frequent prediction in sliding window.
    """
    if not queue:
        return None

    return Counter(queue).most_common(1)[0][0]

# ------------- UI Helpers -----------------

def draw_glass_panel(
    frame: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    alpha: float = 0.25,
) -> None:
    """Draw a semi-transparent glass-style rectangle on the frame.

    This is used as a background panel for HUD elements (text, confidence).
    The effect is created using alpha blending between the original frame
    and an overlay rectangle.
    """
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (25, 25, 25), -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def draw_progress_bar(
    frame: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    value: float,
) -> None:
    """Draw a dynamic progress bar representing model confidence.

    The bar length is proportional to `value` (0–1).
    The color transitions from red (low confidence) to green (high confidence).
    """
    ratio = min(max(value, 0.0), 1.0)
    fill_w = int(w * ratio)

    cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 50), -1)
    color = (0, int(255 * ratio), int(255 * (1 - ratio)))
    cv2.rectangle(frame, (x, y), (x + fill_w, y + h), color, -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (180, 180, 180), 1)
