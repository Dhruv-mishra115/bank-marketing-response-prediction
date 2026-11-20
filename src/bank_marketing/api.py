import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import sklearn
from pydantic import BaseModel, ConfigDict

from .predict import predict_one
from .preprocessing import project_root


def load_artifact(path: Path) -> dict:
    # Joblib uses pickle internally: load only artifacts created by this project.
    import joblib

    artifact = joblib.load(path)
    if not isinstance(artifact, dict) or artifact.get("artifact_format") != 1:
        raise ValueError("Unsupported model artifact format")
    required = {"model", "feature_names", "numeric_features", "categorical_features", "threshold"}
    if not required.issubset(artifact):
        raise ValueError("Model artifact is incomplete")
    version = artifact.get("scikit_learn_version")
    if version is not None and version != sklearn.__version__:
        raise ValueError("Model artifact requires a matching scikit-learn version")
    if not 0 <= float(artifact["threshold"]) <= 1:
        raise ValueError("Model artifact threshold is invalid")
    schema = set(artifact["feature_names"])
    if schema != set(artifact["numeric_features"]) | set(artifact["categorical_features"]):
        raise ValueError("Model artifact feature schema is inconsistent")
    return artifact


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    features: dict[str, Any]


def create_app(artifact_path: str | Path | None = None, metrics_path: str | Path | None = None) -> FastAPI:
    configured_path = Path(artifact_path) if artifact_path else Path(
        os.environ.get("BANK_MARKETING_MODEL_PATH", project_root() / "models" / "selected_model.joblib")
    )
    configured_metrics_path = Path(metrics_path) if metrics_path else Path(
        os.environ.get("BANK_MARKETING_METRICS_PATH", project_root() / "reports" / "dashboard_metrics.json")
    )
    cors_origins = [
        origin.strip() for origin in os.environ.get(
            "BANK_MARKETING_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
        ).split(",") if origin.strip()
    ]

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            app.state.model_artifact = load_artifact(configured_path)
        except FileNotFoundError:
            app.state.model_artifact = None
        yield

    app = FastAPI(
        title="Bank Marketing Response Prediction API",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware, allow_origins=cors_origins, allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"], allow_headers=["Content-Type"],
    )

    @app.get("/healthz")
    def health_check():
        loaded = getattr(app.state, "model_artifact", None) is not None
        if not loaded:
            raise HTTPException(status_code=503, detail="Prediction model is not available")
        return {"status": "ok", "model_loaded": True}

    @app.get("/dashboard/metrics")
    def dashboard_metrics():
        try:
            return json.loads(configured_metrics_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail="Dashboard metrics are not available") from exc
        except (json.JSONDecodeError, OSError) as exc:
            raise HTTPException(status_code=503, detail="Dashboard metrics could not be loaded") from exc

    @app.post("/predict")
    def predict(request: PredictionRequest):
        artifact = getattr(app.state, "model_artifact", None)
        if artifact is None:
            raise HTTPException(status_code=503, detail="Prediction model is not available")
        try:
            result = predict_one(artifact, request.features)
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return result

    return app


app = create_app()
