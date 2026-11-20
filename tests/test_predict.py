import unittest

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from bank_marketing.predict import predict_one
from bank_marketing.preprocessing import build_preprocessor


class PredictionTests(unittest.TestCase):
    def setUp(self):
        X = pd.DataFrame({"age": [20, 30, 40, 50], "job": ["a", "b", "a", "b"]})
        y = pd.Series(["no", "yes", "no", "yes"])
        preprocessor, numeric, categorical = build_preprocessor(X)
        model = Pipeline([("preprocessor", preprocessor), ("classifier", DummyClassifier(strategy="most_frequent"))])
        model.fit(X, y)
        self.artifact = {"model": model, "feature_names": list(X.columns), "numeric_features": numeric, "categorical_features": categorical, "threshold": 0.5}

    def test_valid_prediction(self):
        result = predict_one(self.artifact, {"age": 25, "job": "new-category"})
        self.assertIn(result["predicted_class"], {"no", "yes"})
        self.assertTrue(0 <= result["subscription_probability"] <= 1)

    def test_missing_feature_is_rejected(self):
        with self.assertRaises(ValueError):
            predict_one(self.artifact, {"age": 25})

    def test_invalid_numeric_value_is_rejected(self):
        with self.assertRaises(ValueError):
            predict_one(self.artifact, {"age": float("nan"), "job": "a"})


if __name__ == "__main__":
    unittest.main()
