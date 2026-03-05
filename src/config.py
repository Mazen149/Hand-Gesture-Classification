import os
import joblib


# --------------------------- Project Paths --------------------------
# This ensures paths work correctly regardless of where the script is run.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Output directory for processed videos (inside project root, accessible via Docker volume)
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "videos")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------- Model & Encoder Files --------------------------
# Paths to trained ML artifacts used during inference.

MODEL_PATH = os.path.join(
    BASE_DIR, "models", "HandGesture_XGBoost_Shallow.joblib"
)

ENCODER_PATH = os.path.join(
    BASE_DIR, "models", "label_encoder.joblib"
)

TASK_MODEL_PATH = os.path.join(
    BASE_DIR, "models", "hand_landmarker.task"
)


# --------------------------- Inference Configuration --------------------------
# Parameters controlling gesture prediction behaviour.

# Number of recent predictions used for stabilizing the final gesture label.
PREDICTION_WINDOW = 15

# Minimum confidence required to accept a prediction.
# Predictions below this threshold will be marked as "Uncertain".
CONFIDENCE_THRESHOLD = 0.5


# --------------------------- Hand Landmark Connections --------------------------
# Defines how Mediapipe hand landmarks are connected when drawing
# the skeletal hand structure on frames.

HAND_CONNECTIONS = [
    (5, 9), (9, 13), (13, 17),
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]


# --------------------------- Camera Configuration --------------------------
# Default resolution used when capturing frames from the webcam.

CAMERA_FRAME_WIDTH = 1100
CAMERA_FRAME_HEIGHT = 800


# --------------------------- Video Processing Configuration --------------------------
# Settings used when processing uploaded videos.

# Target resolution for processed videos.
VIDEO_TARGET_WIDTH = 1920
VIDEO_TARGET_HEIGHT = 1080

# Default FPS used if the original video FPS cannot be detected.
VIDEO_DEFAULT_FPS = 30

# Video codec used when writing output videos.
VIDEO_CODEC = "mp4v"