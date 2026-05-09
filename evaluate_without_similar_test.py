from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parent
SIMILAR_PATH = ROOT / "models" / "outputs" / "ionic_26_features_random" / "similar_test_train_pairs.csv"
BEST_OUTPUT_DIR = ROOT / "models" / "outputs" / "ionic_26_features_random_clipped_1e-10"
MODELS = ["lightgbm", "random_forest", "decision_tree", "mlp", "ngboost"]


def metric_row(y_true, y_pred) -> dict[str, float]:
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": mean_squared_error(y_true, y_pred) ** 0.5,
        "r2": r2_score(y_true, y_pred),
    }


def main() -> None:
    similar = pd.read_csv(SIMILAR_PATH)
    similar_test_ids = set(similar["test_id"])
    rows = []
    for model in MODELS:
        predictions = pd.read_csv(BEST_OUTPUT_DIR / model / "test_predictions.csv")
        filtered = predictions[~predictions["ID"].isin(similar_test_ids)]
        removed = predictions[predictions["ID"].isin(similar_test_ids)]
        original_metrics = metric_row(predictions["y_true"], predictions["y_pred"])
        filtered_metrics = metric_row(filtered["y_true"], filtered["y_pred"])
        removed_metrics = metric_row(removed["y_true"], removed["y_pred"])
        rows.append({
            "model": model,
            "original_n": len(predictions),
            "removed_similar_n": len(removed),
            "kept_n": len(filtered),
            "orig_mae": original_metrics["mae"],
            "orig_rmse": original_metrics["rmse"],
            "orig_r2": original_metrics["r2"],
            "filtered_mae": filtered_metrics["mae"],
            "filtered_rmse": filtered_metrics["rmse"],
            "filtered_r2": filtered_metrics["r2"],
            "removed_mae": removed_metrics["mae"],
            "removed_rmse": removed_metrics["rmse"],
            "removed_r2": removed_metrics["r2"],
        })

    output = pd.DataFrame(rows)
    output_path = BEST_OUTPUT_DIR / "filtered_no_similar_test_metrics.csv"
    output.to_csv(output_path, index=False)
    print(f"Wrote {output_path}")
    print(output.to_string(index=False))


if __name__ == "__main__":
    main()
