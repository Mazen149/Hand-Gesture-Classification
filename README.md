# Hand Gesture Classification 🖐️🤖

A real-time Machine Learning application that detects and classifies hand gestures using a webcam. The project utilizes Google's **MediaPipe** for robust hand tracking and landmark extraction, and an **XGBoost Classifier** for high-accuracy, low-latency prediction.

## 🌟 Features

*   **Real-Time Inference:** Fast, live webcam classification with a custom Heads-Up Display (HUD) showing the predicted gesture and confidence scores.
*   **Streamlit Web UI:** An interactive browser-based dashboard with realtime webcam streaming (via WebRTC) and drag-and-drop video upload for batch processing.
*   **MediaPipe Integration:** Efficient hand landmark detection that extracts key spatial features.
*   **XGBoost Classifier:** A trained shallow XGBoost tree model designed for quick inference.
*   **Experiment Benchmarking:** Extensive benchmarking of ML models within Jupyter Notebooks.
*   **Flexible Inference Modes:** Perform hand gesture inference either in real-time using your webcam or offline by providing a pre-recorded video file.
*   **Configurable Parameters:** Adjust confidence threshold and prediction stabilisation window on the fly through the Streamlit sidebar.

## 🧠 How it Works

1. **Landmark Extraction:** By loading the `hand_landmarker.task` module provided by MediaPipe, the inference pipeline pulls $X$ and $Y$ coordinates for 21 hand landmarks.
2. **Normalization:** The `normalize_hand_xy_inference` function takes pixel coordinates and normalizes them so the model performs consistently regardless of the hand's location in the frame.
3. **Classification:** The generated 1D feature array is fed into a loaded `XGBoost` model via joblib, producing categorical predictions.
4. **Stabilization:** A prediction queue (length=15) stabilizes predictions by taking the most frequent classification, reducing frame-to-frame flickering.

## 📦 Dependencies

The core modules used in this project are:
*   `mediapipe==0.10.32`
*   `opencv-python==4.13.0.92`
*   `xgboost==3.2.0`
*   `scikit-learn==1.8.0`
*   `streamlit==1.55.0`
*   `streamlit-webrtc==0.64.5`
*   `av==16.1.0`
*   `pandas`, `numpy`, `matplotlib`, `joblib`

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

### 1. Streamlit Web App (Recommended)
Launch the interactive browser-based dashboard:
```sh
streamlit run app/streamlit/streamlit_app.py
```
*   **Realtime Webcam** – streams your webcam feed via WebRTC with live gesture overlays.
*   **Upload Video** – drag-and-drop a video file for batch processing, then preview and download the annotated result.
*   Adjust **Confidence Threshold** and **Stabilisation Window** from the sidebar.

### 2. CLI – Real-Time Inference (Webcam)
To start the webcam application and launch the hand gesture classifier in real-time:
```sh
python app/cli/realtime_inference.py
```
*   The application displays a live feed with detected hand skeleton, predicted gesture, and confidence score.
*   **Press `ESC`** to cleanly exit the camera feed.

### 3. CLI – Video File Inference
To process a pre-recorded video file and save the predictions:
```sh
python app/cli/video_inference.py
```
*   The script will prompt you to enter the full path to your input video.
*   The processed video with annotations will be saved as `<video_name>_prediction.mp4` in the `output/videos/` directory.
*   Useful for batch processing and creating demonstration videos.



## 🎬 Demo

Below is a demonstration of the model in action, showing real-time hand gesture recognition with the confidence score displayed:



https://github.com/user-attachments/assets/c2fde907-a55b-4b3e-beda-2f5799564ad8



## 🔄 Reproducibility

A core focus of this repository is ensuring all model training and validation can be perfectly reproduced:

1. **Data Preprocessing & Benchmarking Engine:** Navigate to the `notebooks` directory and launch Jupyter Notebook to run the comprehensive pipeline visually.
   ```sh
   jupyter notebook notebooks/data_preparation_and_model_benchmarking.ipynb
   ```
2. **Deterministic Modeling:** Random states are seeded internally to ensure that all generated models (Logistic Regression, SVM, KNN, Random Forest, XGBoost) yield the exact same metrics and splits when instantiated.
3. **Hyperparameter Provenance:** All historical experiments are tracked so researchers can inspect the `mlruns/` directory to fetch the exact hyperparameters, scalers, and label encoders used for our finalized XGBoost model. 
4. **End-to-End Pipeline Validation:** The `.ipynb` encompasses fetching data, scaling/normalizing landmarks, conducting grid-search cross-validation, and logging the metrics iteratively.

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
│       └── ui_utils.py               # CSS injection & display constants
├── data/                              # Dataset directory
├── figures/                           # Generated plots and confusion matrices
├── models/
│   ├── hand_landmarker.task           # MediaPipe pre-trained model for hand tracking
│   ├── HandGesture_XGBoost_Shallow.joblib  # Trained XGBoost model
│   └── label_encoder.joblib           # Scikit-learn LabelEncoder
├── notebooks/
│   └── data_preparation_and_model_benchmarking.ipynb  # Data exploration & model training
├── output/
│   └── videos/                        # Processed video output directory
├── src/                               # Source code for utility modules
│   ├── config.py                      # Centralized configuration and constants
│   ├── inference_utils.py             # Drawing utilities, progress bar, feature extraction
│   ├── metrics.py                     # Evaluation logic
│   ├── preprocessing.py              # Transformations applied to landmark data
│   ├── train.py                       # Training loops and cross-validation
│   └── visualization.py              # Auxiliary visualization scripts
├── requirements.txt                   # Python dependencies
└── README.md                          # Project documentation
```

## ✨ Dataset Overview

The models within this repository were trained and evaluated on a custom-processed dataset boasting the following composition:
*   **Total Observations:** 25,675 annotated samples.
*   **Dimensionality:** 63 features per sample (consisting of 21 MediaPipe hand landmarks extracted across 3 structural axes).
*   **Target Scope:** 18 distinct categorical hand gestures.
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
