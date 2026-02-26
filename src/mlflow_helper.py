from __future__ import annotations

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from typing import Any, Mapping
from matplotlib.figure import Figure
from mlflow.models.signature import ModelSignature


# =========================
# Experiment
# =========================
def set_experiment(name: str, tracking_uri: str | None = None) -> None:
    """Set or create MLflow experiment."""
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(name)

def start_run(run_name: str | None = None) -> mlflow.ActiveRun:
    """Start MLflow run."""
    return mlflow.start_run(run_name=run_name)

def end_run() -> None:
    """End MLflow run."""
    mlflow.end_run()


# =========================
# Logging
# =========================
def log_param(key: str, value: Any) -> None:
    """Log single parameter safely."""
    mlflow.log_param(key, value)

def log_params(params: Mapping[str, Any]) -> None:
    """Log parameters."""
    mlflow.log_params(params)

def log_metrics(metrics: Mapping[str, float]) -> None:
    """Log metrics."""
    mlflow.log_metrics(metrics)

def log_artifact(local_path: str, artifact_path: str | None = None) -> None:
    """Log any local file as an artifact."""
    mlflow.log_artifact(local_path, artifact_path=artifact_path)

def log_text_artifact(text: str, filename: str) -> None:
    """Log text artifact."""
    mlflow.log_text(text, filename)

def log_figure(fig: Figure, filename: str) -> None:
    """Log matplotlib figure."""
    mlflow.log_figure(fig, filename)

def log_model(
    model: Any,
    artifact_path: str = "model",
    signature: ModelSignature | None = None,
    input_example: Any | None = None,
) -> None:
    """
    Log sklearn model with optional signature inference.
    """
    # auto infer if not provided
    if signature is None and input_example is not None:
        preds = model.predict(input_example)
        signature = infer_signature(input_example, preds)

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path=artifact_path,
        signature=signature,
        input_example=input_example,
    )


# =========================
# Tags
# =========================
def set_tags(tags: Mapping[str, str]) -> None:
    """Set MLflow tags."""
    mlflow.set_tags(tags)

def set_tag(key: str, value: str) -> None:
    """Set single MLflow tag."""
    mlflow.set_tag(key, value)