import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluate import evaluate_probabilities
from .models import build_candidates
from .preprocessing import EXCLUDED_FEATURES, chronological_split, load_dataset, project_root


def build_dashboard_metrics(root: Path | None = None, include_threshold_curve: bool = True) -> dict:
    root = Path(root) if root is not None else project_root()
    data_path = root / "data" / "raw" / "bank-additional-full.csv"
    reports = root / "reports"
    frame = load_dataset(data_path)
    target_counts = frame["y"].value_counts()
    predictors = frame.drop(columns=["y"])
    selected = predictors.drop(columns=list(EXCLUDED_FEATURES))
    numeric = selected.select_dtypes(include="number").columns.tolist()
    categorical = selected.select_dtypes(exclude="number").columns.tolist()
    unknown = (frame.select_dtypes(include="str").eq("unknown").sum())
    missing = int(frame.isna().sum().sum())
    monthly = (
        frame.groupby("month", sort=False)["y"]
        .agg(records="size", subscribers=lambda values: int((values == "yes").sum()))
        .reset_index()
    )
    monthly["subscription_rate"] = monthly["subscribers"] / monthly["records"]

    comparison_path = reports / "validation_model_comparison.csv"
    with comparison_path.open(encoding="utf-8", newline="") as stream:
        comparison = list(csv.DictReader(stream))
    for item in comparison:
        for key in ("roc_auc", "average_precision", "precision", "recall", "f1", "accuracy", "brier_score", "actual_subscriber_rate", "mean_predicted_probability", "threshold"):
            item[key] = float(item[key])
        item["confusion_matrix"] = json.loads(item["confusion_matrix"])
    comparison.sort(key=lambda item: item["average_precision"], reverse=True)
    final_metrics = json.loads((reports / "final_test_metrics.json").read_text(encoding="utf-8"))
    importance_path = reports / "feature_importance.csv"
    importance_frame = pd.read_csv(importance_path)
    feature_importance = [
        {"feature": row["feature"], "importance": float(row["importance"]), "interpretation": row["interpretation"]}
        for row in importance_frame.to_dict("records")
    ]

    splits = chronological_split(frame)
    threshold_curve = []
    if include_threshold_curve:
        candidate = build_candidates(splits.X_train)["random_forest_balanced"]
        candidate.fit(splits.X_train, splits.y_train)
        probability = candidate.predict_proba(splits.X_validation)[:, list(candidate.classes_).index("yes")]
        for threshold in np.linspace(0.0, 1.0, 41):
            metrics = evaluate_probabilities(splits.y_validation, probability, float(threshold))
            threshold_curve.append({
                "threshold": float(threshold), "precision": metrics["precision"],
                "recall": metrics["recall"], "f1": metrics["f1"],
                "predicted_subscriber_rate": metrics["predicted_subscriber_rate"],
            })

    category_options = {
        column: sorted(str(value) for value in splits.X_train[column].dropna().unique())
        for column in categorical
    }
    numeric_ranges = {
        column: {"min": float(splits.X_train[column].min()), "max": float(splits.X_train[column].max()), "median": float(splits.X_train[column].median())}
        for column in numeric
    }
    sample = splits.X_train.iloc[0].to_dict()
    sample_profile = {
        key: (value.item() if hasattr(value, "item") else value)
        for key, value in sample.items()
    }
    return {
        "project": {
            "name": "Bank Marketing Response Prediction",
            "description": "Experimental estimate of term-deposit subscription before a marketing call.",
            "status": "Experimental model; not approved for real customer targeting.",
            "dataset_source": "UCI Machine Learning Repository: Bank Marketing (bank-additional-full.csv)",
            "dataset_url": "https://archive.ics.uci.edu/dataset/222/bank%2Bmarketing",
            "target": "y", "positive_class": "yes",
            "excluded_features": list(EXCLUDED_FEATURES),
        },
        "dataset": {
            "records": int(len(frame)), "predictor_count": int(len(predictors.columns)),
            "selected_feature_count": int(len(selected.columns)),
            "numeric_feature_count": len(numeric), "categorical_feature_count": len(categorical),
            "target_counts": {"yes": int(target_counts.get("yes", 0)), "no": int(target_counts.get("no", 0))},
            "target_rates": {"yes": float((frame["y"] == "yes").mean()), "no": float((frame["y"] == "no").mean())},
            "missing_cells": missing, "duplicate_rows": int(frame.duplicated().sum()),
            "unknown_counts": {str(key): int(value) for key, value in unknown[unknown > 0].items()},
            "monthly_rates": monthly.to_dict(orient="records"),
            "numeric_features": numeric, "categorical_features": categorical,
            "category_options": category_options, "numeric_ranges": numeric_ranges,
            "sample_profile": sample_profile,
        },
        "split": {
            "training": {"rows": len(splits.y_train), "subscriber_rate": float((splits.y_train == "yes").mean())},
            "validation": {"rows": len(splits.y_validation), "subscriber_rate": float((splits.y_validation == "yes").mean())},
            "final_segment": {"rows": len(splits.y_test), "subscriber_rate": float((splits.y_test == "yes").mean()), "status": "previously_exposed"},
            "order": "Original UCI file row order (documented as date ordered); no random shuffle",
        },
        "model": {
            "selected_candidate": "random_forest_balanced", "threshold": 0.5,
            "selection_basis": "Highest validation Average Precision among tested candidates; diagnostic only",
            "validation_precision_floor": 0.5,
            "validation_precision_floor_met": False,
            "validation_best_precision_with_positive_predictions": 0.1826,
            "validation_comparison": comparison,
            "threshold_curve": threshold_curve,
            "final_segment_diagnostic": final_metrics,
            "feature_importance": feature_importance,
            "calibration_note": "Balanced Random Forest validation mean predicted probability was about 0.47 vs 0.127 observed; unweighted and chronological sigmoid-calibrated variants did not meet the operating objective.",
        },
        "schema": {
            "feature_names": list(selected.columns), "numeric_features": numeric,
            "categorical_features": categorical, "excluded_features": list(EXCLUDED_FEATURES),
        },
    }


def write_dashboard_metrics(root: Path | None = None) -> Path:
    root = Path(root) if root is not None else project_root()
    destination = root / "reports" / "dashboard_metrics.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = build_dashboard_metrics(root)
    destination.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return destination


if __name__ == "__main__":
    print(write_dashboard_metrics())
