from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "y"
EXCLUDED_FEATURES = ("duration", "campaign")


@dataclass(frozen=True)
class DataSplits:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_validation: pd.DataFrame
    y_validation: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series


def project_root() -> Path:
    source_root = Path(__file__).resolve().parents[2]
    if (source_root / "pyproject.toml").is_file():
        return source_root
    # In a non-editable install, this module lives in site-packages. Render runs
    # the service from the repository root, where the model and reports reside.
    return Path.cwd().resolve()


def load_dataset(path: str | Path | None = None) -> pd.DataFrame:
    data_path = Path(path) if path is not None else project_root() / "data" / "raw" / "bank-additional-full.csv"
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    frame = pd.read_csv(data_path, sep=";")
    if TARGET not in frame.columns:
        raise ValueError(f"Dataset must contain target column {TARGET!r}")
    return frame


def select_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    missing = [column for column in (TARGET, *EXCLUDED_FEATURES) if column not in frame.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    features = frame.drop(columns=[TARGET, *EXCLUDED_FEATURES]).copy()
    target = frame[TARGET].copy()
    return features, target


def chronological_split(
    frame: pd.DataFrame, train_fraction: float = 0.64, validation_fraction: float = 0.16
) -> DataSplits:
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("Split fractions must be between zero and one")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Training and validation fractions must leave records for final test")
    X, y = select_features(frame)
    train_end = int(len(frame) * train_fraction)
    validation_end = int(len(frame) * (train_fraction + validation_fraction))
    if min(train_end, validation_end - train_end, len(frame) - validation_end) < 1:
        raise ValueError("Each chronological segment must contain at least one record")
    return DataSplits(
        X.iloc[:train_end].copy(), y.iloc[:train_end].copy(),
        X.iloc[train_end:validation_end].copy(), y.iloc[train_end:validation_end].copy(),
        X.iloc[validation_end:].copy(), y.iloc[validation_end:].copy(),
    )


def build_preprocessor(X_train: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    numeric_features = X_train.select_dtypes(include="number").columns.tolist()
    categorical_features = X_train.select_dtypes(exclude="number").columns.tolist()
    transformers = []
    if numeric_features:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("num", numeric_pipeline, numeric_features))
    if categorical_features:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ])
        transformers.append(("cat", categorical_pipeline, categorical_features))
    if not transformers:
        raise ValueError("At least one predictor column is required")
    return ColumnTransformer(transformers), numeric_features, categorical_features
