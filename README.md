# Hand Gesture Classification 🖐️🤖

A real-time Machine Learning application that detects and classifies hand gestures using a webcam. The project utilizes Google's **MediaPipe** for robust hand tracking and landmark extraction, and an **XGBoost Classifier** for high-accuracy, low-latency prediction.

## 🌟 Features

*   **Real-Time Inference:** Fast, live webcam classification with a custom Heads-Up Display (HUD) showing the predicted gesture and confidence scores.
*   **MediaPipe Integration:** Efficient hand landmark detection that extracts key spatial features.
*   **XGBoost Classifier:** A trained shallow XGBoost tree model designed for quick inference.
*   **MLflow Tracking:** Comprehensive experiment tracking, including model hyperparameters and evaluation metrics.
*   **Experiment Benchmarking:** Extensive benchmarking of ML models within Jupyter Notebooks.
*   **Video Recording:** The application can locally record the inference sessions for demonstrations.

## 🧠 How it Works

1. **Landmark Extraction:** By loading the `hand_landmarker.task` module provided by MediaPipe, `app/inference.py` pulls $X$ and $Y$ coordinates for 21 hand landmarks.
2. **Normalization:** The `normalize_hand_xy_inference` function takes pixel coordinates and normalizes them so the model performs consistently regardless of the hand's location in the frame.
3. **Classification:** The generated 1D feature array is fed into a loaded `XGBoost` model via joblib, producing categorical predictions.
4. **Stabilization:** A prediction queue (length=15) stabilizes predictions by taking the most frequent classification, reducing frame-to-frame flickering.

## 📦 Dependencies

The core modules used in this project are:
*   `mediapipe==0.10.32`
*   `opencv-python` / `opencv==4.13.0`
*   `xgboost==3.2.0`
*   `scikit-learn==1.8.0`
*   `mlflow==3.10.0`
*   `pandas`, `numpy`, `matplotlib`

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

### 1. Real-Time Inference (Running the App)
To start the webcam application and launch the hand gesture classifier, run the inference script:
```sh
python app/inference.py
```
*   The script will prompt you: `Do you want to save the video? (y/n):`
*   Select `y` if you want to generate an `output_recording.mp4` file of your session.
*   **Press `ESC`** to cleanly exit the camera feed.

### 2. Viewing MLflow Experiments
If you want to view the stored MLflow tracking configurations and metrics:
```sh
mlflow ui
```
Then navigate to `http://127.0.0.1:5000/` in your browser.


## 🔄 Reproducibility

A core focus of this repository is ensuring all model training and validation can be perfectly reproduced:

1. **Data Preprocessing & Benchmarking Engine:** Navigate to the `notebooks` directory and launch Jupyter Notebook to run the comprehensive pipeline visually.
   ```sh
   jupyter notebook notebooks/data_preparation_and_model_benchmarking.ipynb
   ```
2. **Deterministic Modeling:** Random states are seeded internally to ensure that all generated models (Logistic Regression, SVM, KNN, Random Forest, XGBoost) yield the exact same metrics and splits when instantiated.
3. **Hyperparameter Provenance:** Because all historical experiments are tracked with `MLflow`, researchers can easily inspect the `mlruns/` directory to fetch the exact hyperparameters, scalers, and label encoders used for our finalized XGBoost model. 
4. **End-to-End Pipeline Validation:** The `.ipynb` encompasses fetching data, scaling/normalizing landmarks, conducting grid-search cross-validation, and subsequently logging the metrics to MLflow iteratively.

## 📂 Project Structure

```text
Hand-Gesture-Classification/
├── app/
│   └── inference.py          # Main application script to run real-time webcam inference
├── data/                     # Dataset directory
├── figures/                  # Generated plots and confusion matrices
├── mlflow screenshots/       # Visual evidence of MLflow tracking setup
├── mlruns/                   # MLflow experiment tracking logs
├── models/
│   ├── hand_landmarker.task  # MediaPipe pre-trained model for hand tracking
│   ├── HandGesture_XGBoost_Shallow.joblib # Trained XGBoost model
│   └── label_encoder.joblib  # Scikit-learn LabelEncoder
├── notebooks/
│   └── data_preparation_and_model_benchmarking.ipynb  # Data exploration and model training pipeline
├── src/                      # Source code for utility scripts
│   ├── inference_utils.py    # Drawing utilities, progress bar, feature extraction
│   ├── metrics.py            # Evaluation logic
│   ├── mlflow_helper.py      # MLflow setup & utilities
│   ├── mlflow_logging.py     # End-to-end model logging function
│   ├── preprocessing.py      # Transformations applied to landmark data
│   ├── train.py              # Training loops and cross-validation
│   └── visualization.py      # Auxiliary visualization scripts
└── requirements.txt          # Python dependencies
```

## ✨ Dataset Overview

The models within this repository were trained and evaluated on a custom-processed dataset boasting the following composition:
*   **Total Observations:** 25,675 annotated samples.
*   **Dimensionality:** 63 features per sample (consisting of 21 MediaPipe hand landmarks extracted across 3 structural axes).
*   **Target Scope:** 18 distinct categorical hand gestures.
*   **Recognized Gestures:** ![alt text](image-1.png)

## 📈 Experiment Tracking & Model Benchmarking

This project leverages **MLflow** to track, compare, and benchmark various machine learning models (KNN, SVM, Logistic Regression, Random Forest, and XGBoost). 

### 1. Model Metrics Overview
We tracked model performance (Accuracy, F1-Score, Precision, and Recall) across all hyperparameters to identify the most optimal classifier.
<br>
<img src="mlflow%20screentshots/Comparing%20all%20models.png" width="800">

### 2. General Experiments
A snapshot of all executed runs logged locally through MLflow.
<br>
<img src="mlflow%20screentshots/Expirement.png" width="800">

### 3. Best Performing Models
Runs dynamically sorted by the `test_f1_weighted` metric, highlighting XGBoost's superior performance.
<br>
<img src="mlflow%20screentshots/Expirement%20sorted%20by%20f1.png" width="800">

### 4. Hyperparameter Analysis
A parallel coordinates plot visually analyzing hyperparameter combinations across the top 4 experiment runs.
<br>
<img src="image (1).png" width="800">

### 5. Model Registry
The best performing models are registered via MLflow's Model Registry with explicit lifecycle stages (e.g. Production, Staging).

**Production Model: XGBoost**
<br>
<img src="mlflow%20screentshots/Registered%20XGBoost.png" width="800">

**Staging Model: Random Forest**
<br>
<img src="mlflow%20screentshots/Registered%20Random%20Forest.png" width="800">

### 📊 Model Comparison
Here is a comprehensive comparison of all the trained models ranking their performance across Training, Validation, and Test metrics.

![alt text](image.png)

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
