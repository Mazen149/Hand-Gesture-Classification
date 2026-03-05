# ── Stage 1: Builder ────────────────────────────────────────────
# Install Python packages in an isolated prefix so only the final
# artefacts are copied to the runtime image (no build caches/cruft).
FROM python:3.10-slim AS builder

WORKDIR /build

COPY requirements.txt ./

# Build a Docker-optimised requirements on-the-fly:
#   • opencv-python  →  opencv-python-headless  (no GUI libs needed, ~100 MB smaller)
#   • matplotlib removed                        (unused by Streamlit app, ~50 MB smaller)
RUN sed \
    -e 's/opencv-python==/opencv-python-headless==/' \
    -e '/^matplotlib/d' \
    requirements.txt > requirements-docker.txt \
    && pip install --no-cache-dir --prefix=/install -r requirements-docker.txt


# ── Stage 2: Runtime ───────────────────────────────────────────
FROM python:3.10-slim

WORKDIR /app

# Only the minimal system libs actually needed at runtime:
#   • libglib2.0-0  – required by MediaPipe / OpenCV
#   • ffmpeg        – H.264 re-encoding for uploaded-video playback
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    ffmpeg \
    && apt-get purge -y --auto-remove -o APT::AutoRemove::RecommendsImportant=false \
    && rm -rf /var/lib/apt/lists/*

# Copy pre-installed Python packages from the builder stage
COPY --from=builder /install /usr/local

# Copy project files (filtered by .dockerignore)
COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app/streamlit/streamlit_app.py"]