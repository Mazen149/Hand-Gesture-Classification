"""
Streamlit frontend for Hand Gesture Classification.

Two modes:
  1. Realtime Webcam  – runs continuously until the user clicks Stop.
  2. Video Inference  – upload a video, get an annotated output to download.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".." , "..")))


import streamlit as st
from src.config import CONFIDENCE_THRESHOLD, PREDICTION_WINDOW
from app.streamlit.ui_utils import inject_css, DISPLAY_WIDTH, SUPPORTED_GESTURES_IMAGE_URL
from app.streamlit.model_utils import load_artifacts
from app.streamlit.pages import page_realtime, page_video

st.set_page_config(
    page_title="Hand Gesture AI",
    page_icon="🖐️",
    layout="wide",
    initial_sidebar_state="expanded",
)

def main():
    inject_css()

    # ── Sidebar Settings ──
    with st.sidebar:
        st.markdown("## ⚙️ Configurations")
        st.markdown("Adjust inference parameters in real-time.")
        user_conf_threshold = st.slider(
            "Confidence Threshold", 
            min_value=0.0, 
            max_value=1.0, 
            value=float(CONFIDENCE_THRESHOLD), 
            step=0.05,
            help="Minimum confidence score required to lock in a gesture prediction."
        )
        user_pred_window = st.number_input(
            "Stabilisation Window (frames)", 
            min_value=1, 
            max_value=60, 
            value=int(PREDICTION_WINDOW), 
            step=1,
            help="Number of consecutive frames used to smooth out flickering predictions."
        )

    # ── Hero Section ──
    st.markdown(
        '<div class="hero">'
        "<h1>🖐️ Hand Gesture AI</h1>"
        "<p>Real-time tracking and video classification pipeline powered by Machine Learning</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    model, encoder = load_artifacts()

    # ── Supported Gestures (Image safely inside the container box) ──
    classes = list(getattr(encoder, "classes_", []))
    st.markdown(
        f'<div class="mode-card" style="padding: 1.2rem;">'
        f'<h4 style="margin-top:0; color:#0f172a;">📚 Supported Gestures Map ({len(classes)} Classes)</h4>'
        f'<p style="color:#64748b; font-size: 0.9rem; margin-bottom: 1rem;">Visual reference for all gesture definitions actively being classified by the model.</p>'
        f'<img src="{SUPPORTED_GESTURES_IMAGE_URL}" alt="Supported Gestures Map" style="width: 100%; border-radius: 12px; border: 1px solid #e2e8f0; display: block; margin: 0 auto;">'
        f'</div>',
        unsafe_allow_html=True
    )
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Mode Selection ──
    st.markdown("### 🚦 Select Inference Mode")
    mode = st.segmented_control(
        "Choose Inference Pipeline",
        options=["📹 Realtime Webcam", "🎬 Upload Video"],
        default=None,
        key="inference_mode",
        label_visibility="collapsed"
    )

    if mode:
        st.divider()
        if mode == "📹 Realtime Webcam":
            page_realtime(model, encoder, user_conf_threshold, user_pred_window)
        elif mode == "🎬 Upload Video":
            page_video(model, encoder, user_conf_threshold, user_pred_window)

if __name__ == "__main__":
    main()