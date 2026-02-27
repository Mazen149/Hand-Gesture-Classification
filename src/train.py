from typing import Any, Dict, List, Type

import os
import textwrap
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from .metrics import compute_metrics


# Base project directory (one level above src/)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _build_param_suffix(params: Dict[str, Any], max_items: int = 4) -> str:
    """
    Build a short, generic suffix from model parameters.

    Example:
        n_estimators=300, max_depth=10
        -> nest300_maxd10
    """

    parts = []

    # sort for stability
    for i, (k, v) in enumerate(sorted(params.items())):
        if v is None:
            continue
        if i >= max_items:  # prevent very long names
            break

        short_key = k[:5].lower()
        short_val = str(v).replace(" ", "")
        parts.append(f"{short_key}{short_val}")

    return "_".join(parts) if parts else "default"


def train_and_evaluate_models(
    model_class: Type[Any],
    param_grid: List[Dict[str, Any]],
    model_name: str,
    X_train: Any,
    y_train: Any,
    X_validation: Any,
    y_validation: Any,
    X_test: Any,
    y_test: Any,
    encoder: Any,
) -> List[Dict[str, Any]]:
    """
    Train and evaluate a model over a parameter grid.

    For each parameter set:
    - trains the model
    - computes train/validation/test metrics
    - plots and saves confusion matrix
    - stores all results for later MLflow logging

    Returns:
        List of dictionaries containing model, params, metrics, and artifacts.
    """

    results: List[Dict[str, Any]] = []

    for params in param_grid:
        model = model_class(**params)

        # Train
        model.fit(X_train, y_train)

        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_val = model.predict(X_validation)
        y_pred_test = model.predict(X_test)

        # Metrics
        metrics_train = compute_metrics(y_train, y_pred_train)
        metrics_val = compute_metrics(y_validation, y_pred_val)
        metrics_test = compute_metrics(y_test, y_pred_test)

        # Print
        print("\n" + "=" * 110)
        print(f"{model_name} | Params:", params)

        print("\n--- Train Metrics ---")
        print(metrics_train)

        print("\n--- Validation Metrics ---")
        print(metrics_val)

        print("\n--- Test Metrics (FINAL) ---")
        print(metrics_test)

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred_test)

        fig, ax = plt.subplots(figsize=(8, 8))

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=getattr(encoder, "classes_", None),
        )

        disp.plot(
            cmap="viridis",
            values_format="d",
            ax=ax,
            colorbar=True,
        )

        # Wrap params text
        params_str = str(params)
        wrapped_params = "\n".join(textwrap.wrap(params_str, width=50))

        ax.set_title(
            f"{model_name}\nParams:\n{wrapped_params}",
            fontsize=14,
            pad=20,
        )

        ax.tick_params(axis="x", labelrotation=90, labelsize=10)
        ax.tick_params(axis="y", labelsize=10)

        plt.tight_layout()

        # Save figure
        safe_name = model_name.lower().replace(" ", "_")

        param_suffix = _build_param_suffix(params)
        param_suffix = (
            param_suffix
            .replace(".", "p")
            .replace("-", "")
            .replace(":", "")
            .replace("/", "")
        )

        filename = f"{safe_name}_cm_test_{param_suffix}.png"

        FIGURES_DIR = os.path.join(BASE_DIR, "figures")
        os.makedirs(FIGURES_DIR, exist_ok=True)

        filename = os.path.join(FIGURES_DIR, filename)

        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.show()

        # Store results
        results.append(
            {
                "model_name": model_name,
                "params": params,
                "model": model,
                "metrics_train": metrics_train,
                "metrics_val": metrics_val,
                "metrics_test": metrics_test,
                "pred_test": y_pred_test,
                "confusion_matrix": cm,
                "cm_path": filename,
            }
        )

    return results