from typing import Any, Mapping
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

def compute_metrics(
    y_true: Any,
    y_pred: Any
) -> Mapping[str, float]:
    """
    Compute standard weighted classification metrics.

    Returns accuracy, weighted precision, weighted recall,
    and weighted F1-score for the given predictions.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_weighted": precision_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "recall_weighted": recall_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "f1_weighted": f1_score(
            y_true, y_pred, average="weighted"
        ),
    }