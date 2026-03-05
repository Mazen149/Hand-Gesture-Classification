import streamlit as st

# ── Display Defaults ────────────────────────────────────────────────────────
DISPLAY_WIDTH = 960  
SUPPORTED_GESTURES_IMAGE_URL = "https://github.com/user-attachments/assets/94880cd3-08cb-438b-92cf-dd1c409411a5"
WEBCAM_TARGET_WIDTH = 720
WEBCAM_TARGET_HEIGHT = 340

# ── Custom CSS ──────────────────────────────────────────────────────────────
def inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
        /* ...existing CSS from streamlit_app.py... */
        </style>
        """,
        unsafe_allow_html=True,
    )
