import streamlit as st
import av
import cv2
import os
import threading
from collections import deque
from pathlib import Path

# ── Monkey-patch for streamlit-webrtc on Python 3.14+ ──────────────
# Version 0.64.5 crashes when _polling_thread is None (fixed on GitHub
# main but not yet released).  Guard the stop() method so it handles None.
import streamlit_webrtc.shutdown as _sw

_orig_stop = _sw.SessionShutdownObserver.stop

def _patched_stop(self, timeout: float = 1.0):
    if self._polling_thread is None:
        return
    _orig_stop(self, timeout)

_sw.SessionShutdownObserver.stop = _patched_stop
# ────────────────────────────────────────────────────────────────────

from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import mediapipe as mp
from .model_utils import create_landmarker
from .video_utils import annotate_frame, _process_video
from .ui_utils import WEBCAM_TARGET_WIDTH, WEBCAM_TARGET_HEIGHT
from src.inference_utils import (
    extract_landmarks, get_stable_prediction, normalize_hand_xy_inference,
    draw_glass_panel, draw_progress_bar,
)
from src.config import HAND_CONNECTIONS, OUTPUT_DIR


class GestureProcessor:
    """Thread-safe state holder for the WebRTC callback.

    All mutable state lives here (not in st.session_state) so the
    video-frame callback thread can safely read/write without
    touching Streamlit internals.
    """

    def __init__(self, model, encoder, conf_threshold, pred_window):
        self.model = model
        self.encoder = encoder
        self.conf_threshold = conf_threshold
        self.pred_queue: deque = deque(maxlen=pred_window)
        self.landmarker = create_landmarker()
        self.timestamp = 0
        self.lock = threading.Lock()
        # Shared latest result for the main thread to display
        self.last_label = "No Hand"
        self.last_confidence = 1.0

    def update_settings(self, conf_threshold, pred_window):
        """Update settings from the UI sliders (called on every Streamlit rerun)."""
        with self.lock:
            threshold_changed = self.conf_threshold != conf_threshold
            self.conf_threshold = conf_threshold
            if self.pred_queue.maxlen != pred_window:
                # Resize by creating a new deque, preserving recent predictions
                old = list(self.pred_queue)
                self.pred_queue = deque(old, maxlen=pred_window)
            # Clear the queue so the new threshold takes effect immediately
            # instead of being masked by stale predictions from the old threshold
            if threshold_changed:
                self.pred_queue.clear()

    def process(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w, _ = img.shape

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        with self.lock:
            self.timestamp += 1
            ts = self.timestamp

        try:
            result = self.landmarker.detect_for_video(mp_image, ts)
        except Exception:
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        label_text = "No Hand"
        confidence_value = 1.0

        # Snapshot current settings under the lock so the callback
        # always uses the latest slider values from the main thread.
        with self.lock:
            conf_threshold = self.conf_threshold
            pred_queue = self.pred_queue

        if result.hand_landmarks:
            hand_lm = result.hand_landmarks[0]
            features = extract_landmarks(hand_lm, w, h)
            features_norm = normalize_hand_xy_inference(features)

            pred_encoded = self.model.predict(features_norm)[0]
            pred_label = str(self.encoder.inverse_transform([pred_encoded])[0])

            if hasattr(self.model, "predict_proba"):
                confidence_value = float(
                    self.model.predict_proba(features_norm).max()
                )
                if confidence_value < conf_threshold:
                    pred_label = "Uncertain"

            with self.lock:
                pred_queue.append(pred_label)
                stable = get_stable_prediction(pred_queue)
            if stable:
                label_text = stable

            # Draw hand skeleton
            px_coords = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lm]
            for i, j in HAND_CONNECTIONS:
                cv2.line(img, px_coords[i], px_coords[j], (0, 255, 255), 2)
            for pt in px_coords:
                cv2.circle(img, pt, 4, (255, 0, 255), -1)

        # HUD overlay (compact, equal padding on all sides)
        panel_x, panel_y = 6, 6
        pad = 14
        panel_w = 210
        text_x = panel_x + pad
        gesture_y = panel_y + pad + 14
        conf_y = gesture_y + 20
        bar_y = conf_y + 12
        bar_w = panel_w - 2 * pad
        bar_h = 8
        panel_h = (bar_y + bar_h + pad) - panel_y

        draw_glass_panel(img, panel_x, panel_y, panel_w, panel_h, 1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, f"Gesture: {label_text}",
                    (text_x, gesture_y), font, 0.42, (0, 255, 0), 1)
        cv2.putText(img, f"Confidence: {confidence_value * 100:.1f}%",
                    (text_x, conf_y), font, 0.36, (0, 255, 0), 1)
        draw_progress_bar(img, text_x, bar_y, bar_w, bar_h, confidence_value)

        # Store for the main thread
        with self.lock:
            self.last_label = label_text
            self.last_confidence = confidence_value

        return av.VideoFrame.from_ndarray(img, format="bgr24")


def page_realtime(model, encoder, conf_threshold, pred_window):
    st.markdown(
        '<div class="mode-card"><h3>📹 Realtime Camera Feed</h3>'
        '<p>Your webcam feed runs continuously with live gesture predictions. '
        'Click the <b>Fullscreen Icon</b> on the video to expand the feed! '
        'Dashboard stats are embedded.</p></div>',
        unsafe_allow_html=True,
    )

    # Create processor once and cache it in session_state
    if "gesture_processor" not in st.session_state:
        st.session_state.gesture_processor = GestureProcessor(
            model, encoder, conf_threshold, pred_window
        )
    processor: GestureProcessor = st.session_state.gesture_processor
    # Sync slider values into the processor so changes take effect live
    processor.update_settings(conf_threshold, pred_window)

    # WebRTC configuration
    rtc_configuration = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    webrtc_ctx = webrtc_streamer(
        key="hand-gesture-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_configuration,
        media_stream_constraints={
            "video": {
                "width": {"ideal": WEBCAM_TARGET_WIDTH},
                "height": {"ideal": WEBCAM_TARGET_HEIGHT},
            },
            "audio": False,
        },
        async_processing=True,
        video_frame_callback=processor.process,
    )

    if webrtc_ctx.state.playing:
        st.success("🎥 **Live Feed Active!** Camera is streaming.")
    else:
        st.info("💡 Click **START** above, then allow camera access in the browser.")

def page_video(model, encoder, conf_threshold, pred_window):
    st.markdown(
        '<div class="mode-card"><h3>🎬 Batch Video Processing</h3>'
        '<p>Upload a pre-recorded video file. The ML pipeline will process and annotate every frame. '
        'Once complete, you can preview and download the finished product.</p></div>',
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "Drag and drop your video here",
        type=["mp4", "avi", "mov", "mkv", "webm"],
        help="Supported formats: MP4, AVI, MOV, MKV, WEBM",
    )

    if uploaded is None:
        return

    # Use configured output directory accessible from Docker volumes
    input_path = os.path.join(OUTPUT_DIR, uploaded.name)

    with open(input_path, "wb") as f:
        f.write(uploaded.read())

    stem = Path(uploaded.name).stem
    output_path = os.path.join(OUTPUT_DIR, f"{stem}_prediction.mp4")

    col_preview, col_action = st.columns([3, 1], vertical_alignment="top")

    with col_preview:
        with st.expander("👀 Preview Original Upload", expanded=False):
            st.video(input_path)

    with col_action:
        if st.button("🚀 Start Inference", type="primary", use_container_width=True):
            _process_video(
                input_path,
                output_path,
                model,
                encoder,
                conf_threshold,
                pred_window,
            )
            st.session_state["video_output_path"] = output_path
            st.session_state["video_output_name"] = Path(output_path).name

    output_path_state = st.session_state.get("video_output_path")
    output_name_state = st.session_state.get("video_output_name", "prediction.mp4")

    if output_path_state and os.path.exists(output_path_state):
        st.subheader("✅ Processing Complete")
        col_result, col_download = st.columns([2, 1], gap="large")
        with col_result:
            with open(output_path_state, "rb") as f:
                st.video(f.read())
        with col_download:
            st.info(
                "Your annotated video is ready! The original frame rate and aspect ratio have been preserved."
            )
            with open(output_path_state, "rb") as f:
                st.download_button(
                    label="⬇ Download Annotated Video",
                    data=f,
                    file_name=output_name_state,
                    mime="video/mp4",
                    type="primary",
                    use_container_width=True,
                )
