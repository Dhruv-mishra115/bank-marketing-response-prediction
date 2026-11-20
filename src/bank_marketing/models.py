from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from .preprocessing import build_preprocessor


def _pipeline(X_train, classifier):
    preprocessor, _, _ = build_preprocessor(X_train)
    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])


def build_candidates(X_train) -> dict:
    """Return a small, reproducible candidate set; all preprocessing fits inside each pipeline."""
    candidates = {
        "logistic_regression": _pipeline(X_train, LogisticRegression(max_iter=1000)),
        "class_weighted_logistic": _pipeline(
            X_train, LogisticRegression(max_iter=1000, class_weight="balanced")
        ),
        "decision_tree": _pipeline(
            X_train, DecisionTreeClassifier(
                max_depth=5, min_samples_split=100, min_samples_leaf=50,
                class_weight="balanced", random_state=42
            )
        ),
    }
    rf_settings = [
        ("random_forest_balanced", "balanced", 10, 50),
        ("random_forest_unweighted", None, 10, 50),
        ("random_forest_balanced_shallower", "balanced", 8, 50),
        ("random_forest_balanced_larger_leaf", "balanced", 10, 100),
    ]
    for name, class_weight, depth, leaf in rf_settings:
        candidates[name] = _pipeline(
            X_train, RandomForestClassifier(
                n_estimators=200, max_depth=depth, min_samples_split=100,
                min_samples_leaf=leaf, class_weight=class_weight,
                random_state=42, n_jobs=-1
            )
        )
    # Calibration folds are chronological and use only the model-training segment.
    rf_for_calibration = _pipeline(
        X_train, RandomForestClassifier(
            n_estimators=100, max_depth=10, min_samples_split=100,
            min_samples_leaf=50, class_weight="balanced",
            random_state=42, n_jobs=-1
        )
    )
    candidates["calibrated_random_forest"] = CalibratedClassifierCV(
        estimator=rf_for_calibration, method="sigmoid",
        cv=TimeSeriesSplit(n_splits=3), ensemble=True
    )
    return candidates
