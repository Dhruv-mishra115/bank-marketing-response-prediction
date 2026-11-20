import numpy as np
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)


def evaluate_probabilities(y_true, probabilities, threshold: float = 0.5) -> dict:
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between zero and one")
    actual = np.asarray(y_true) == "yes"
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 1 or len(probabilities) != len(actual):
        raise ValueError("y_true and probabilities must be matching one-dimensional arrays")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("probabilities must be finite values between zero and one")
    predicted = probabilities >= threshold
    return {
        "roc_auc": float(roc_auc_score(actual, probabilities)),
        "average_precision": float(average_precision_score(actual, probabilities)),
        "precision": float(precision_score(actual, predicted, zero_division=0)),
        "recall": float(recall_score(actual, predicted, zero_division=0)),
        "f1": float(f1_score(actual, predicted, zero_division=0)),
        "accuracy": float(accuracy_score(actual, predicted)),
        "brier_score": float(brier_score_loss(actual, probabilities)),
        "actual_subscriber_rate": float(actual.mean()),
        "mean_predicted_probability": float(probabilities.mean()),
        "predicted_subscriber_rate": float(predicted.mean()),
        "confusion_matrix": confusion_matrix(actual, predicted, labels=[False, True]).tolist(),
        "threshold": float(threshold),
    }
