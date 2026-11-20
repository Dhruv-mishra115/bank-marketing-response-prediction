import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from .evaluate import evaluate_probabilities
from .models import build_candidates
from .preprocessing import chronological_split, load_dataset

LOGGER = logging.getLogger("bank_marketing")


def compare_validation_models(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict, object]:
    splits = chronological_split(frame)
    candidates = build_candidates(splits.X_train)
    records = []
    fitted = {}
    for name, model in candidates.items():
        LOGGER.info("Fitting %s on %d training rows", name, len(splits.X_train))
        model.fit(splits.X_train, splits.y_train)
        classes = list(model.classes_)
        yes_index = classes.index("yes")
        probabilities = model.predict_proba(splits.X_validation)[:, yes_index]
        metrics = evaluate_probabilities(splits.y_validation, probabilities)
        records.append({"model": name, **{key: value for key, value in metrics.items() if key != "confusion_matrix"},
                        "confusion_matrix": json.dumps(metrics["confusion_matrix"])})
        fitted[name] = model
    results = pd.DataFrame(records).sort_values(
        ["average_precision", "roc_auc"], ascending=False
    ).reset_index(drop=True)
    return results, fitted, splits


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare bank marketing models on the chronological validation segment.")
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    frame = load_dataset(args.data)
    results, _, splits = compare_validation_models(frame)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "validation_model_comparison.csv"
    results.to_csv(output_path, index=False)
    print(results.drop(columns="confusion_matrix").to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nValidation results saved to {output_path}")
    print(f"Training prevalence: {(splits.y_train == 'yes').mean():.4%}; validation prevalence: {(splits.y_validation == 'yes').mean():.4%}")
    print("The final test segment was not used by this comparison.")


if __name__ == "__main__":
    main()
