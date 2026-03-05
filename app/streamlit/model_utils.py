import joblib
import mediapipe as mp
from src.config import MODEL_PATH, ENCODER_PATH, TASK_MODEL_PATH
import streamlit as st

@st.cache_resource(show_spinner=False)
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    return model, encoder

def create_landmarker():
    opts = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=TASK_MODEL_PATH),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=1,
    )
    return mp.tasks.vision.HandLandmarker.create_from_options(opts)
