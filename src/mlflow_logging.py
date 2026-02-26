from typing import Any, Dict, List

import pandas as pd

from sklearn.metrics import classification_report

from .train import _build_param_suffix
from . import mlflow_helper

def _build_run_name(model_family: str, params: Dict[str, Any]) -> str:
    """
    Build a compact, generic MLflow run name from model family and parameters.
    """

    # priority params (most informative first)
    priority_keys = [
        "n_estimators",
        "max_depth",
        "learning_rate",
        "C",
        "kernel",
        "n_neighbors",
    ]

    parts = [model_family]

    # add priority params first (if exist)
    for key in priority_keys:
        if key in params and params[key] is not None:
            val = str(params[key]).replace(" ", "")
            parts.append(f"{key[:4]}{val}")

    # fallback: add remaining params (shortened)
    remaining = [
        k for k in params.keys()
        if k not in priority_keys
    ]

    for key in sorted(remaining):
        val = params[key]
        if val is None:
            continue
        short_key = key[:4]
        short_val = str(val).replace(" ", "")
        parts.append(f"{short_key}{short_val}")

    # keep name from becoming huge
    run_name = "_".join(parts)

    # MLflow UI becomes messy with very long names
    return run_name[:120]


def log_runs_to_mlflow(
    results: List[Dict[str, Any]],
    model_family: str,
    X_train: Any,
    X_validation: Any,
    X_test: Any,
    y_test: Any,
) -> None:
    """
    Log trained model runs to MLflow.

    For each run:
    - logs parameters
    - logs test metrics
    - logs classification report
    - logs confusion matrix artifact
    - logs model
    """

    for run_data in results:
        params = run_data["params"]
        model = run_data["model"]
        metrics_test = run_data["metrics_test"]
        y_pred_test = run_data["pred_test"]

        # Generic run
        run_name = _build_run_name(model_family, params)

        with mlflow_helper.start_run(run_name=run_name):

            # Tag
            mlflow_helper.set_tag("model_family", model_family)

            # Params
            mlflow_helper.log_params(params)

            # Metrics
            test_metrics_prefixed = {
                f"test_{k}": v for k, v in metrics_test.items()
            }
            mlflow_helper.log_metrics(test_metrics_prefixed)

            # Classification report
            report = classification_report(y_test, y_pred_test, zero_division=0)
            mlflow_helper.log_text_artifact(
                report,
                "classification_report.txt",
            )

            # Confusion matrix artifact
            mlflow_helper.log_artifact(
                run_data["cm_path"],
                artifact_path="figures",
            )

            # Dataset info
            mlflow_helper.log_param("n_train_samples", len(X_train))
            mlflow_helper.log_param("n_val_samples", len(X_validation))
            mlflow_helper.log_param("n_test_samples", len(X_test))
            mlflow_helper.log_param("n_features", X_train.shape[1])

            # Model logging
            safe_family = model_family.lower().replace(" ", "_")

            # Build short param suffix for model name
            param_suffix = _build_param_suffix(params)
            param_suffix = (
                param_suffix
                .replace(".", "p")
                .replace("-", "")
                .replace(":", "")
                .replace("/", "")
            )

            model_artifact_path = f"{safe_family}_model_{param_suffix}"

            input_example = (
                X_train.iloc[:1]
                if hasattr(X_train, "iloc")
                else pd.DataFrame(X_train[:1])
            )

            mlflow_helper.log_model(
                model=model,
                artifact_path=model_artifact_path,
                input_example=input_example,
            )

    print(f"\n\n[[[{model_family} runs logged to MLflow]]]")