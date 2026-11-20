from numbers import Real

import numpy as np
import pandas as pd


def predict_one(artifact: dict, features: dict) -> dict:
    expected = artifact["feature_names"]
    missing = sorted(set(expected) - set(features))
    extra = sorted(set(features) - set(expected))
    if missing or extra:
        raise ValueError(f"Feature schema mismatch (missing={missing}, extra={extra})")
    numeric = set(artifact["numeric_features"])
    categorical = set(artifact["categorical_features"])
    row = {}
    for name in expected:
        value = features[name]
        if value is None:
            row[name] = np.nan
        elif name in numeric:
            if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
                raise ValueError(f"Feature {name!r} must be a finite number or null")
            row[name] = value
        elif name in categorical:
            if not isinstance(value, str):
                raise ValueError(f"Feature {name!r} must be a string or null")
            row[name] = value
        else:
            raise ValueError(f"Feature {name!r} is not in the saved schema")
    frame = pd.DataFrame([row], columns=expected)
    model = artifact["model"]
    classes = list(model.classes_)
    yes_index = classes.index("yes")
    probability = float(model.predict_proba(frame)[0, yes_index])
    threshold = float(artifact["threshold"])
    return {
        "predicted_class": "yes" if probability >= threshold else "no",
        "subscription_probability": probability,
        "threshold": threshold,
    }
