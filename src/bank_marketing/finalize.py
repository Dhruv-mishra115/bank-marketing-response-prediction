import argparse
import hashlib
import json
import logging
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.metrics import classification_report

from .evaluate import evaluate_probabilities
from .models import build_candidates
from .preprocessing import chronological_split, load_dataset

LOGGER = logging.getLogger("bank_marketing")


def feature_importance(model, feature_names):
    classifier = model.named_steps["classifier"]
    if hasattr(classifier, "feature_importances_"):
        values = classifier.feature_importances_
        kind = "tree_importance"
    elif hasattr(classifier, "coef_"):
        values = abs(classifier.coef_[0])
        kind = "absolute_logistic_coefficient"
    else:
        return None
    names = model.named_steps["preprocessor"].get_feature_names_out()
    return pd.DataFrame({"feature": names, "importance": values, "interpretation": kind}).sort_values(
        "importance", ascending=False
    )


def run_final_evaluation(data_path=None, model_name="random_forest_balanced", threshold=0.5,
                         output_dir=Path("."), test_status="previously_exposed"):
    frame = load_dataset(data_path)
    splits = chronological_split(frame)
    X_development = pd.concat([splits.X_train, splits.X_validation], axis=0)
    y_development = pd.concat([splits.y_train, splits.y_validation], axis=0)
    candidates = build_candidates(X_development)
    if model_name not in candidates:
        raise ValueError(f"Unknown model candidate: {model_name}")
    model = candidates[model_name]
    LOGGER.info("Fitting selected candidate %s on training + validation (%d rows)", model_name, len(X_development))
    model.fit(X_development, y_development)
    yes_index = list(model.classes_).index("yes")
    probabilities = model.predict_proba(splits.X_test)[:, yes_index]
    metrics = evaluate_probabilities(splits.y_test, probabilities, threshold)
    predicted_labels = ["yes" if value >= threshold else "no" for value in probabilities]
    metrics["classification_report"] = classification_report(
        splits.y_test, predicted_labels, labels=["no", "yes"], output_dict=True, zero_division=0
    )
    metrics.update({
        "selected_model": model_name,
        "training_rows": len(splits.X_train),
        "validation_rows": len(splits.X_validation),
        "final_test_rows": len(splits.X_test),
        "threshold_source": "default_0.5; business constraint not met on validation candidates",
        "test_status": test_status,
    })
    root = Path(output_dir)
    models_dir = root / "models"
    reports_dir = root / "reports"
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    feature_names = list(X_development.columns)
    numeric = X_development.select_dtypes(include="number").columns.tolist()
    categorical = X_development.select_dtypes(exclude="number").columns.tolist()
    data_file = Path(data_path) if data_path else Path(__file__).resolve().parents[2] / "data" / "raw" / "bank-additional-full.csv"
    fingerprint = hashlib.sha256(data_file.read_bytes()).hexdigest()
    artifact = {
        "artifact_format": 1, "model": model, "model_name": model_name,
        "feature_names": feature_names, "numeric_features": numeric,
        "categorical_features": categorical, "excluded_features": ["duration", "campaign"],
        "threshold": float(threshold), "threshold_source": metrics["threshold_source"],
        "test_status": test_status, "data_sha256": fingerprint,
        "scikit_learn_version": sklearn.__version__,
    }
    artifact_path = models_dir / "selected_model.joblib"
    joblib.dump(artifact, artifact_path)
    (reports_dir / "final_test_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    importance = feature_importance(model, feature_names)
    if importance is not None:
        importance.to_csv(reports_dir / "feature_importance.csv", index=False)
    return metrics, artifact_path


def main():
    parser = argparse.ArgumentParser(description="Fit a selected model on train + validation, then evaluate once on the final chronological segment.")
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--model", default="random_forest_balanced")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--test-status", choices=["previously_exposed", "new_untouched"], default="previously_exposed")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    metrics, artifact_path = run_final_evaluation(
        args.data, args.model, args.threshold, args.output_dir, args.test_status
    )
    print(json.dumps(metrics, indent=2))
    print(f"\nModel artifact saved to {artifact_path}")


if __name__ == "__main__":
    main()
