# Hand Gesture Classification 🖐️🤖

> **An End-to-End Machine Learning Project** — from raw data exploration and feature engineering, through model training and experiment tracking, to a fully **Dockerized** and **deployed** production application.

![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-3.10.0-0194E2?logo=mlflow&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)
[![Docker Hub](https://img.shields.io/badge/Docker%20Hub-Published-2496ED?logo=docker&logoColor=white)](https://hub.docker.com/repository/docker/mazen1393/hand-gesture-streamlit/general)
![Streamlit](https://img.shields.io/badge/Streamlit-1.55.0-FF4B4B?logo=streamlit&logoColor=white)
![Streamlit Cloud](https://img.shields.io/badge/Deployed-Streamlit%20Cloud-FF4B4B?logo=streamlit&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-3.2.0-189FDD?logo=xgboost&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.32-00A98F)
![OpenCV](https://img.shields.io/badge/OpenCV-4.13.0-5C3EE8?logo=opencv&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7.2-F7931E?logo=scikitlearn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

A real-time Machine Learning application that detects and classifies **18 hand gestures** using a webcam. The project covers the **complete ML lifecycle**: data preparation, feature normalization, benchmarking **5 model families** with **14 experiments** tracked in **MLflow**, and deploying the best model through a **Dockerized Streamlit web app** with live WebRTC streaming.

### 🔑 Project Highlights

|                                  |                                                                                                                                                      |
| -------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| 🏗️ **End-to-End Pipeline**        | Data exploration → Preprocessing → Training → Evaluation → MLflow Tracking → Deployment                                                              |
| 🐳 **Dockerized**                 | Multi-stage Docker build published on [my Docker Hub repository](https://hub.docker.com/repository/docker/mazen1393/hand-gesture-streamlit/general), |
| ☁️ **Deployed**                   | Live on [Streamlit Cloud](https://hand-gesture-classification-webapp.streamlit.app/) — zero setup required                                           |
| 📊 **MLflow Experiment Tracking** | 14 runs across 5 model families, with parameters, metrics, artifacts, and model registration                                                         |
| 🎯 **98.17% Test F1**             | XGBoost Shallow model selected for production after rigorous benchmarking                                                                            |

---

## 🌟 Features

*   **Real-Time Inference:** Fast, live webcam classification with a custom Heads-Up Display (HUD) showing the predicted gesture and confidence scores.
*   **Streamlit Web UI:** An interactive browser-based dashboard with realtime webcam streaming (via WebRTC) and drag-and-drop video upload for batch processing.
*   **MediaPipe Integration:** Efficient hand landmark detection that extracts 21 key spatial landmarks per frame.
*   **XGBoost Classifier:** A trained shallow XGBoost tree model (`max_depth=6`) designed for ultra-fast inference.
*   **MLflow Experiment Tracking:** All training runs are logged with parameters, metrics, classification reports, confusion matrices, and serialized models.
*   **Model Registry:** Best models registered in MLflow with aliases (`Production` for XGBoost, `Staging` for Random Forest).
*   **Experiment Benchmarking:** Extensive benchmarking of 5 ML model families (14 configurations) within Jupyter Notebooks.
*   **Flexible Inference Modes:** Perform hand gesture inference either in real-time using your webcam or offline by providing a pre-recorded video file.
*   **Configurable Parameters:** Adjust confidence threshold and prediction stabilisation window on the fly through the Streamlit sidebar.
*   **Dockerized Deployment:** Multi-stage Docker build for a lightweight, production-ready container image.

## 🧠 How it Works

```
📹 Webcam Frame
    │
    ▼
🖐️ MediaPipe Hand Landmarker ──► 21 landmarks (x, y, z) ──► 63D feature vector
    │
    ▼
📐 Normalization (translate to wrist origin + scale by palm size)
    │
    ▼
🤖 XGBoost Classifier ──► Gesture Label + Confidence Score
    │
    ▼
🔄 Prediction Stabilization (sliding window majority vote)
    │
    ▼
🖥️ HUD Overlay (glass panel + progress bar + label)
```

1. **Landmark Extraction:** The `hand_landmarker.task` MediaPipe module detects 21 hand landmarks, producing a 63-dimensional feature vector ($21 \times 3$ coordinates).
2. **Normalization:** Landmarks are translated to the wrist origin and scaled by the wrist-to-middle-finger distance, ensuring position and scale invariance.
3. **Classification:** The normalized feature array is fed into the production XGBoost model via joblib, producing gesture predictions with confidence scores.
4. **Stabilization:** A configurable prediction queue (default length=15) stabilizes predictions via majority voting, eliminating frame-to-frame flickering.

## 📦 Dependencies

The core modules used in this project are:
*   `numpy==2.2.6`
*   `pandas==2.3.3`
*   `matplotlib==3.10.8`
*   `scikit-learn==1.7.2`
*   `xgboost==3.2.0`
*   `mediapipe==0.10.32`
*   `opencv-python-headless==4.13.0.92`
*   `mlflow==3.10.0`
*   `joblib==1.5.3`
*   `streamlit==1.55.0`
*   `streamlit-webrtc==0.64.5`
*   `av==16.1.0`

## 🛠️ Installation

1. **Clone the repository:**
   ```sh
   git clone https://github.com/Mazen149/Hand-Gesture-Classification.git
   cd Hand-Gesture-Classification
   ```

2. **Create a virtual environment (Optional but Recommended):**
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

## 🚀 Usage


### 1. Streamlit Cloud (Recommended)

The app is deployed on **Streamlit Cloud** and can be accessed directly without any installation:

👉 **[Live Demo – Hand Gesture Classification](https://hand-gesture-classification-webapp.streamlit.app/)**

*   **Realtime Webcam** – streams your webcam feed via WebRTC with live gesture overlays.
*   **Upload Video** – drag-and-drop a video file for batch processing, then preview and download the annotated result.
*   Adjust **Confidence Threshold** and **Stabilisation Window** from the sidebar.

### 2. Streamlit Web App (Local)
To run the Streamlit web app on your local machine:
```sh
streamlit run app/streamlit/streamlit_app.py
```

### 3. Docker

You can run the pre-built image directly from **[my Docker Hub repository](https://hub.docker.com/repository/docker/mazen1393/hand-gesture-streamlit/general)**:
```sh
docker run -p 8501:8501 mazen1393/hand-gesture-streamlit:1.0
```


Or build and run the Streamlit app inside a container using **Docker Compose** (make sure to run this command in the directory containing your `docker-compose.yml` file):
```sh
docker compose up --build
```

The app will be available at `http://localhost:8501`.

> **Docker Build Details:**  The multi-stage Dockerfile uses `python:3.10-slim` as the base. The builder stage swaps `opencv-python` → `opencv-python-headless` (~100 MB savings) and strips `matplotlib` (~50 MB savings). The runtime stage includes only `libglib2.0-0` (for MediaPipe/OpenCV) and `ffmpeg` (for H.264 video re-encoding), resulting in a lean production image.


### 4. Command Line Interface (CLI)

#### a) Real-Time Inference (Webcam)
To start the webcam application and launch the hand gesture classifier in real-time:
```sh
python app/cli/realtime_inference.py
```
*   The application displays a live feed with detected hand skeleton, predicted gesture, and confidence score.
*   **Press `ESC`** to cleanly exit the camera feed.

#### b) Video File Inference
To process a pre-recorded video file and save the predictions:
```sh
python app/cli/video_inference.py
```
*   The script will prompt you to enter the full path to your input video.
*   The processed video with annotations will be saved as `<video_name>_prediction.mp4` in the `output/videos/` directory.
*   Useful for batch processing and creating demonstration videos.

### 5. Viewing MLflow Experiments
To explore all logged training runs, metrics, and artifacts locally, launch the MLflow UI:
```sh
mlflow ui --backend-store-uri file:mlruns
```
Then open `http://localhost:5000` in your browser to browse experiments, compare runs, and inspect registered models.


## 🎬 Demo

Below is a demonstration of the model in action, showing real-time hand gesture recognition with the confidence score displayed:



https://github.com/user-attachments/assets/e9928064-89bb-435e-b062-3bb6ba8bd3f6



## 📊 MLflow Experiment Tracking

All model training runs are tracked in **MLflow**, providing full reproducibility and transparency. The experiment `Hand_Gesture_Classification` contains **14 runs** across **5 model families**, each logging:

*   **Parameters:** All hyperparameters, dataset sizes (`n_train_samples`, `n_val_samples`, `n_test_samples`), and feature count.
*   **Metrics:** Test accuracy, F1-weighted, precision-weighted, and recall-weighted.
*   **Artifacts:** Confusion matrix plots, classification reports, and serialized model files.
*   **Tags:** Model family identifier for easy filtering.

### Experiment Dashboard

The full experiment view shows all 14 runs sorted by test F1-score:

<p align="center">
  <img src="mlflow screentshots/Expirement sorted by f1.png" alt="MLflow Runs Sorted by F1" width="100%"/>
</p>

### Model Comparison

Side-by-side comparison of all models and a focused view of the top 4 performers:

<p align="center">
  <img src="mlflow screentshots/Comparing all models.png" alt="Comparing All Models" width="100%"/>
</p>

<p align="center">
  <img src="mlflow screentshots/Comparing best 4 models.png" alt="Comparing Best 4 Models" width="100%"/>
</p>

### Registered Models

The two best-performing models were registered in the **MLflow Model Registry** with aliases:

| Model                                                   | Alias          | Use Case                     |
| ------------------------------------------------------- | -------------- | ---------------------------- |
| XGBoost Shallow (`max_depth=6`, `n_estimators=200`)     | **Production** | Live inference (low latency) |
| Random Forest Deep (`max_depth=30`, `n_estimators=300`) | **Staging**    | Backup / further evaluation  |

<p align="center">
  <img src="mlflow screentshots/Registered XGBoost.png" alt="Registered XGBoost Model" width="100%"/>
</p>

<p align="center">
  <img src="mlflow screentshots/Registered Random Forest.png" alt="Registered Random Forest Model" width="100%"/>
</p>

### Per-Model Run Examples

<details>
<summary><b>XGBoost Runs (2 configurations)</b></summary>
<br>

**XGBoost Config 1** — `n_estimators=200, max_depth=6, learning_rate=0.1` (Production Model)
<p align="center">
  <img src="mlflow screentshots/XGBoost/XGBoost 1 Overivew.png" alt="XGBoost 1 Overview" width="100%"/>
  <img src="mlflow screentshots/XGBoost/XGBoost 1 Model Metrics.png" alt="XGBoost 1 Metrics" width="100%"/>
</p>

**XGBoost Config 2** — `n_estimators=400, max_depth=8, learning_rate=0.05`
<p align="center">
  <img src="mlflow screentshots/XGBoost/XGBoost 2 Overivew.png" alt="XGBoost 2 Overview" width="100%"/>
  <img src="mlflow screentshots/XGBoost/XGBoost 2 Model Metrics.png" alt="XGBoost 2 Metrics" width="100%"/>
</p>

</details>

<details>
<summary><b>Random Forest Runs (3 configurations)</b></summary>
<br>

**Random Forest Config 1** — `n_estimators=100, max_depth=None`
<p align="center">
  <img src="mlflow screentshots/Random Forest/Random Forest 1 Overview.png" alt="RF 1 Overview" width="100%"/>
  <img src="mlflow screentshots/Random Forest/Random Forest 1 Model Metrics.png" alt="RF 1 Metrics" width="100%"/>
</p>

**Random Forest Config 2** — `n_estimators=300, max_depth=30` (Staging Model)
<p align="center">
  <img src="mlflow screentshots/Random Forest/Random Forest 2 Overview.png" alt="RF 2 Overview" width="100%"/>
  <img src="mlflow screentshots/Random Forest/Random Forest 2 Model Metrics.png" alt="RF 2 Metrics" width="100%"/>
</p>

**Random Forest Config 3** — `n_estimators=500, min_samples_split=5`
<p align="center">
  <img src="mlflow screentshots/Random Forest/Random Forest 3 Overview.png" alt="RF 3 Overview" width="100%"/>
  <img src="mlflow screentshots/Random Forest/Random Forest 3 Model Metrics.png" alt="RF 3 Metrics" width="100%"/>
</p>

</details>

<details>
<summary><b>SVM Runs (3 configurations)</b></summary>
<br>

**SVM Config 1** — `kernel=rbf, C=1.0, gamma=scale`
<p align="center">
  <img src="mlflow screentshots/SVM/SVM 1 Overview.png" alt="SVM 1 Overview" width="100%"/>
  <img src="mlflow screentshots/SVM/SVM 1 Model Metrics.png" alt="SVM 1 Metrics" width="100%"/>
</p>

**SVM Config 2** — `kernel=rbf, C=10.0, gamma=scale`
<p align="center">
  <img src="mlflow screentshots/SVM/SVM 2 Overview.png" alt="SVM 2 Overview" width="100%"/>
  <img src="mlflow screentshots/SVM/SVM 2 Model metrics.png" alt="SVM 2 Metrics" width="100%"/>
</p>

**SVM Config 3** — `kernel=linear, C=1.0`
<p align="center">
  <img src="mlflow screentshots/SVM/SVM 3 Overview.png" alt="SVM 3 Overview" width="100%"/>
  <img src="mlflow screentshots/SVM/SVM 3 Model Metrics.png" alt="SVM 3 Metrics" width="100%"/>
</p>

</details>

<details>
<summary><b>KNN Runs (3 configurations)</b></summary>
<br>

**KNN Config 1** — `n_neighbors=3, weights=uniform, metric=minkowski`
<p align="center">
  <img src="mlflow screentshots/KNN/KNN 1 Overview.png" alt="KNN 1 Overview" width="100%"/>
  <img src="mlflow screentshots/KNN/KNN 1 Model Metrics.png" alt="KNN 1 Metrics" width="100%"/>
</p>

**KNN Config 2** — `n_neighbors=5, weights=distance, metric=minkowski`
<p align="center">
  <img src="mlflow screentshots/KNN/KNN 2 Overview.png" alt="KNN 2 Overview" width="100%"/>
  <img src="mlflow screentshots/KNN/KNN 2 Model Metrics.png" alt="KNN 2 Metrics" width="100%"/>
</p>

**KNN Config 3** — `n_neighbors=7, weights=distance, metric=minkowski`
<p align="center">
  <img src="mlflow screentshots/KNN/KNN 3 Overview.png" alt="KNN 3 Overview" width="100%"/>
  <img src="mlflow screentshots/KNN/KNN 3 Model Metrics.png" alt="KNN 3 Metrics" width="100%"/>
</p>

</details>

<details>
<summary><b>Logistic Regression Runs (3 configurations)</b></summary>
<br>

**Logistic Regression Config 1** — `C=1.0, penalty=l2, solver=lbfgs`
<p align="center">
  <img src="mlflow screentshots/Logistic Regression/Logistic Regression 1 Overview.png" alt="LR 1 Overview" width="100%"/>
  <img src="mlflow screentshots/Logistic Regression/Logistic Regression 1 Model Metrics.png" alt="LR 1 Metrics" width="100%"/>
</p>

**Logistic Regression Config 2** — `C=0.5, penalty=l2, solver=lbfgs`
<p align="center">
  <img src="mlflow screentshots/Logistic Regression/Logistic Regression 2 Overview.png" alt="LR 2 Overview" width="100%"/>
  <img src="mlflow screentshots/Logistic Regression/Logistic Regression 2 Model Metrics.png" alt="LR 2 Metrics" width="100%"/>
</p>

**Logistic Regression Config 3** — `C=2.0, penalty=l2, solver=lbfgs`
<p align="center">
  <img src="mlflow screentshots/Logistic Regression/Logistic Regression 3 Overview.png" alt="LR 3 Overview" width="100%"/>
  <img src="mlflow screentshots/Logistic Regression/Logistic Regression 3 Model Metrics.png" alt="LR 3 Metrics" width="100%"/>
</p>

</details>

---

## 🔄 Reproducibility

A core focus of this repository is ensuring all model training and validation can be perfectly reproduced:

1. **Data Preprocessing & Benchmarking Engine:** Navigate to the `notebooks` directory and launch Jupyter Notebook to run the comprehensive pipeline visually.
   ```sh
   jupyter notebook notebooks/data_preparation_and_model_benchmarking.ipynb
   ```
2. **MLflow Experiment Tracking:** Every training run is logged to MLflow with full parameter snapshots, evaluation metrics, confusion matrices, and serialized models — enabling exact reproduction and comparison.
3. **Deterministic Modeling:** Random states are seeded internally to ensure that all generated models (Logistic Regression, SVM, KNN, Random Forest, XGBoost) yield the exact same metrics and splits when instantiated.
4. **Hyperparameter Provenance:** Every experiment configuration is preserved inside the benchmarking notebook and MLflow, so researchers can inspect the exact hyperparameter grids, scalers, and label encoders used for the finalized XGBoost model.
5. **End-to-End Pipeline Validation:** The notebook encompasses loading data, scaling/normalizing landmarks, running parameter-grid evaluations across five model families, logging to MLflow, and comparing metrics via a leaderboard table.

## 📂 Project Structure

```text
Hand-Gesture-Classification/
├── app/
│   ├── cli/
│   │   ├── realtime_inference.py      # CLI real-time webcam inference
│   │   └── video_inference.py         # CLI video file batch inference
│   └── streamlit/
│       ├── streamlit_app.py           # Streamlit entry point
│       ├── pages.py                   # Realtime & video upload pages
│       ├── model_utils.py             # Cached model/encoder loading
│       ├── video_utils.py             # Frame annotation & video processing
│       └── ui_utils.py                # CSS injection & display constants
├── data/
│   └── hand_landmarks_data.csv        # 25,675 annotated gesture samples
├── figures/                           # Generated confusion matrix plots (14 files)
├── mlflow screentshots/               # MLflow UI screenshots for documentation
├── mlruns/                            # MLflow local tracking backend
├── models/
│   ├── hand_landmarker.task           # MediaPipe pre-trained hand tracking model
│   ├── HandGesture_XGBoost_Shallow.joblib  # Production XGBoost model
│   ├── HandGesture_RandomForest_Deep.joblib # Staging Random Forest model
│   └── label_encoder.joblib           # Scikit-learn LabelEncoder (18 classes)
├── notebooks/
│   └── data_preparation_and_model_benchmarking.ipynb  # Full training & evaluation pipeline
├── src/                               # Source code for utility modules
│   ├── config.py                      # Centralized configuration and constants
│   ├── inference_utils.py             # Landmark extraction, normalization, HUD drawing
│   ├── metrics.py                     # Weighted accuracy, precision, recall, F1 computation
│   ├── mlflow_helper.py               # MLflow wrapper (experiment, logging, model registry)
│   ├── mlflow_logging.py              # Automated run logging (params, metrics, artifacts)
│   ├── preprocessing.py               # Hand landmark normalization (translation + scaling)
│   ├── train.py                       # Training loops and parameter-grid evaluation
│   └── visualization.py               # Hand skeleton plotting & gesture grid visualization
├── Dockerfile                         # Multi-stage Docker build (Python 3.10-slim)
├── docker-compose.yml                 # One-command container deployment
├── .dockerignore                      # Files excluded from Docker build context
├── requirements.txt                   # Python dependencies
└── README.md                          # Project documentation
```

## ✨ Dataset Overview

The models within this repository were trained and evaluated on a custom-processed dataset boasting the following composition:
*   **Total Observations:** 25,675 annotated samples.
*   **Dimensionality:** 63 features per sample (consisting of 21 MediaPipe hand landmarks extracted across 3 structural axes).
*   **Target Scope:** 18 distinct categorical hand gestures.
*   **Data Split:** 80% Train / 10% Validation / 10% Test (stratified).
*   **Recognized Gestures:** <img width="1400" height="373" alt="image-1" src="https://github.com/user-attachments/assets/94880cd3-08cb-438b-92cf-dd1c409411a5" />


### 📊 Model Comparison
Here is a comprehensive comparison of all the trained models ranking their performance across Training, Validation, and Test metrics.


<img width="1518" height="626" alt="image" src="https://github.com/user-attachments/assets/514aa4d2-83eb-4f68-bcdd-31440307e0ae" />


### 🏆 Final Algorithm Decision: XGBoost (Shallow Tree)
After comprehensive evaluation of multiple algorithms, the **XGBoost (Shallow)** model was selected as our final production model. Based on the benchmarking matrix, it is ranked #1 for our specific use-case.

**Reasons for Selection:**
1. **Top-Tier Performance:** It achieves an outstanding **Test Accuracy and F1-score of 0.9817**, with almost identical Validation metrics (0.9829), proving strong generalization capabilities.
2. **Low Inference Latency (Real-Time Ready):** The configuration utilizes a relatively shallow architecture (`max_depth=6`, `n_estimators=200`). This smaller tree depth translates to ultra-fast inference per frame, absolutely essential for real-time webcam prediction without lag.
3. **Overfitting Mitigation:** Deeper/larger models (like XGBoost with `n_estimators=400` or Random Forests with `max_depth=30`) barely achieved any improvement in Test F1 but incurred significantly more computational overhead and memory. The shallow XGBoost strikes the perfect balance between bias, variance, and latency.


## ⚠️ Important Considerations

While this model demonstrates strong local benchmarking results, there are a few inherent limitations worth noting:
*   **Single Hand Inference:** The current configuration is explicitly tuned for `num_hands=1`. It will only classify the primary tracked hand in the frame.
*   **Lighting Sensitivity:** As with most standard camera-based vision models, very poor lighting conditions may occasionally impact landmark estimation reliability before classification can even occur.

## 📬 Get in Touch

For questions, feedback, or collaboration opportunities, feel free to connect via [LinkedIn](https://www.linkedin.com/in/mazen-mohamed-363371361/).

## 📜 License

This project is licensed under the [MIT License](https://choosealicense.com/licenses/mit/).
