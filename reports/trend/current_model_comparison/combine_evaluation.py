"""Combine general experimental and halide evaluation results."""

from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
MODELS = ["catboost", "random_forest", "xgboost", "lightgbm"]


def prediction_paths(model):
    if model == "catboost":
        return [
            ROOT / "data/experimental/annotations/experimental-data-predict-trend-catboost.csv",
            ROOT / "data/experimental/annotations/halide-predict-trend-catboost.csv",
        ]
    return [OUT / f"experimental-{model}.csv", OUT / f"halide-{model}.csv"]


def main():
    rows = []
    combined_predictions = []
    for model in MODELS:
        frames = []
        for dataset, path in zip(["general_experimental", "halide"], prediction_paths(model)):
            frame = pd.read_csv(path)
            frame.insert(0, "dataset", dataset)
            frames.append(frame)
        combined = pd.concat(frames, ignore_index=True)
        combined.insert(0, "model", model)
        combined_predictions.append(combined)
        true = combined["真实趋势标签"]
        predicted = combined["预测趋势标签"]
        rows.append({
            "model": model,
            "pairs": len(combined),
            "accuracy": accuracy_score(true, predicted),
            "macro_f1": f1_score(true, predicted, average="macro", zero_division=0),
            "balanced_accuracy": balanced_accuracy_score(true, predicted),
            "decrease": (true == "decrease").sum(),
            "unchanged": (true == "unchanged").sum(),
            "increase": (true == "increase").sum(),
        })
    pd.DataFrame(rows).to_csv(OUT / "combined_experimental_metrics.csv", index=False)
    pd.concat(combined_predictions, ignore_index=True).to_csv(OUT / "combined_experimental_predictions.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
