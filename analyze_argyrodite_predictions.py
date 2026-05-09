from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "models" / "outputs" / "argyrodite_target_analysis"
MODELS = ["lightgbm", "random_forest", "decision_tree", "mlp", "ngboost"]
DATASETS = [
    ("fixed", ROOT / "models" / "outputs" / "ionic_26_features"),
    ("fixed clipped", ROOT / "models" / "outputs" / "ionic_26_features_clipped_1e-10"),
    ("random", ROOT / "models" / "outputs" / "ionic_26_features_random"),
    ("random clipped", ROOT / "models" / "outputs" / "ionic_26_features_random_clipped_1e-10"),
]


def metric_values(frame: pd.DataFrame) -> dict[str, float]:
    r2 = r2_score(frame["y_true"], frame["y_pred"]) if len(frame) > 1 else np.nan
    return {
        "mae": mean_absolute_error(frame["y_true"], frame["y_pred"]),
        "rmse": mean_squared_error(frame["y_true"], frame["y_pred"]) ** 0.5,
        "r2": r2,
    }


def main() -> None:
    raw = pd.read_csv(ROOT / "rawdata" / "all.csv")[
        ["ID", "Reduced Composition", "Family", "Ionic conductivity (S cm-1)"]
    ]
    raw["is_argyrodite"] = raw["Family"].astype(str).str.contains("argyro", case=False, na=False)
    raw["is_s_argyrodite"] = (
        raw["is_argyrodite"]
        & raw["Reduced Composition"].astype(str).str.contains(r"S(?![a-z])", regex=True, na=False)
    )

    metric_rows = []
    prediction_rows = []
    for dataset_label, dataset_dir in DATASETS:
        for model in MODELS:
            prediction_path = dataset_dir / model / "test_predictions.csv"
            if not prediction_path.exists():
                continue
            predictions = pd.read_csv(prediction_path).merge(raw, on="ID", how="left")
            for subset_name, subset_mask in {
                "argyrodite": predictions["is_argyrodite"],
                "s_argyrodite": predictions["is_s_argyrodite"],
            }.items():
                subset = predictions[subset_mask].copy()
                if subset.empty:
                    continue
                metrics = metric_values(subset)
                metric_rows.append({
                    "dataset": dataset_label,
                    "subset": subset_name,
                    "model": model,
                    "n_test": len(subset),
                    **metrics,
                })
                for _, row in subset.iterrows():
                    prediction_rows.append({
                        "dataset": dataset_label,
                        "subset": subset_name,
                        "model": model,
                        "ID": row["ID"],
                        "Reduced Composition": row["Reduced Composition"],
                        "Family": row["Family"],
                        "conductivity": row["Ionic conductivity (S cm-1)"],
                        "y_true": row["y_true"],
                        "y_pred": row["y_pred"],
                        "abs_error": abs(row["y_true"] - row["y_pred"]),
                    })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(metric_rows)
    predictions = pd.DataFrame(prediction_rows)
    metrics.to_csv(OUTPUT_DIR / "argyrodite_test_metrics.csv", index=False)
    predictions.to_csv(OUTPUT_DIR / "argyrodite_test_predictions.csv", index=False)
    print(metrics.to_string(index=False))
    print(f"Wrote outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
