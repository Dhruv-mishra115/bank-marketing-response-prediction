import tempfile
import unittest
import json
import os
from pathlib import Path
from unittest.mock import patch

import joblib
import pandas as pd
from fastapi.testclient import TestClient
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from bank_marketing.api import create_app
from bank_marketing.preprocessing import build_preprocessor


class ApiTests(unittest.TestCase):
    def setUp(self):
        X = pd.DataFrame({"age": [20, 30, 40, 50], "job": ["a", "b", "a", "b"]})
        y = pd.Series(["no", "yes", "no", "yes"])
        preprocessor, numeric, categorical = build_preprocessor(X)
        model = Pipeline([("preprocessor", preprocessor), ("classifier", DummyClassifier(strategy="most_frequent"))])
        model.fit(X, y)
        artifact = {"artifact_format": 1, "model": model, "feature_names": list(X.columns), "numeric_features": numeric, "categorical_features": categorical, "threshold": 0.5}
        self.temp_dir = tempfile.TemporaryDirectory()
        self.path = Path(self.temp_dir.name) / "test_model.joblib"
        joblib.dump(artifact, self.path)
        self.metrics_path = Path(self.temp_dir.name) / "dashboard_metrics.json"
        self.metrics_path.write_text(json.dumps({"project": {"name": "test"}}), encoding="utf-8")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_health_and_prediction(self):
        with TestClient(create_app(self.path)) as client:
            self.assertEqual(client.get("/healthz").json()["status"], "ok")
            response = client.post("/predict", json={"features": {"age": 27, "job": "a"}})
            self.assertEqual(response.status_code, 200)
            self.assertIn(response.json()["predicted_class"], {"no", "yes"})

    def test_invalid_schema_is_rejected(self):
        with TestClient(create_app(self.path)) as client:
            response = client.post("/predict", json={"features": {"age": 27}})
            self.assertEqual(response.status_code, 422)

    def test_health_when_artifact_is_missing(self):
        missing = self.path.parent / "missing.joblib"
        with TestClient(create_app(missing)) as client:
            health = client.get("/healthz")
            self.assertEqual(health.status_code, 503)
            response = client.post("/predict", json={"features": {"age": 27, "job": "a"}})
            self.assertEqual(response.status_code, 503)

    def test_dashboard_metrics_endpoint(self):
        with TestClient(create_app(self.path, self.metrics_path)) as client:
            response = client.get("/dashboard/metrics")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["project"]["name"], "test")

    def test_vite_origin_is_allowed_by_cors(self):
        with TestClient(create_app(self.path, self.metrics_path)) as client:
            response = client.options(
                "/predict",
                headers={
                    "Origin": "http://localhost:5173",
                    "Access-Control-Request-Method": "POST",
                    "Access-Control-Request-Headers": "content-type",
                },
            )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers["access-control-allow-origin"], "http://localhost:5173")

    def test_cors_can_be_restricted_to_deployed_frontend_origin(self):
        origin = "https://bank-marketing-dashboard.example.vercel.app"
        with patch.dict(os.environ, {"BANK_MARKETING_CORS_ORIGINS": origin}):
            with TestClient(create_app(self.path, self.metrics_path)) as client:
                allowed = client.options(
                    "/predict",
                    headers={
                        "Origin": origin,
                        "Access-Control-Request-Method": "POST",
                        "Access-Control-Request-Headers": "content-type",
                    },
                )
                self.assertEqual(allowed.status_code, 200)
                self.assertEqual(allowed.headers["access-control-allow-origin"], origin)

                denied = client.options(
                    "/predict",
                    headers={
                        "Origin": "https://untrusted.example",
                        "Access-Control-Request-Method": "POST",
                        "Access-Control-Request-Headers": "content-type",
                    },
                )
                self.assertNotIn("access-control-allow-origin", denied.headers)


if __name__ == "__main__":
    unittest.main()
