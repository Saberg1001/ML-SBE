from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .train import metrics


def plot_lightgbm_feature_evolution(
    runs: list[tuple[str, str | Path]],
    output_path: str | Path,
) -> pd.DataFrame:
    """Compare LightGBM train parity and train/test metrics across feature sets."""
    import matplotlib.pyplot as plt

    loaded = []
    metric_rows = []
    for label, run_dir in runs:
        model_dir = Path(run_dir) / "lightgbm"
        train = pd.read_csv(model_dir / "train_predictions.csv")
        test = pd.read_csv(model_dir / "test_predictions.csv")
        train_metrics = metrics(train["y_true"], train["y_pred"])
        test_metrics = metrics(test["y_true"], test["y_pred"])
        loaded.append((label, train))
        metric_rows.append({
            "Feature set": label,
            "Train MAE": train_metrics["mae"],
            "Test MAE": test_metrics["mae"],
            "Train RMSE": train_metrics["rmse"],
            "Test RMSE": test_metrics["rmse"],
            "Train R²": train_metrics["r2"],
            "Test R²": test_metrics["r2"],
        })

    all_values = np.concatenate([
        frame[["y_true", "y_pred"]].to_numpy().reshape(-1)
        for _, frame in loaded
    ])
    lower = float(np.nanmin(all_values))
    upper = float(np.nanmax(all_values))
    padding = max((upper - lower) * 0.04, 0.1)

    fig = plt.figure(figsize=(18, 11), constrained_layout=True)
    grid = fig.add_gridspec(2, len(loaded), height_ratios=[1.35, 0.8])
    for column, (label, frame) in enumerate(loaded):
        axis = fig.add_subplot(grid[0, column])
        axis.scatter(
            frame["y_true"], frame["y_pred"], s=38, alpha=0.72,
            color="cornflowerblue", edgecolor="royalblue", linewidth=0.5,
        )
        axis.plot([lower, upper], [lower, upper], "--", color="#e52421", linewidth=2)
        axis.set_xlim(lower - padding, upper + padding)
        axis.set_ylim(lower - padding, upper + padding)
        axis.set_title(label, fontsize=18, fontweight="bold")
        axis.set_xlabel("True log10(conductivity)", fontsize=13)
        if column == 0:
            axis.set_ylabel("Predicted log10(conductivity)", fontsize=13)
        axis.grid(alpha=0.25)

    metric_frame = pd.DataFrame(metric_rows)
    table_axis = fig.add_subplot(grid[1, :])
    table_axis.axis("off")
    headers = list(metric_frame.columns)
    cells = [
        [row["Feature set"], *[f"{row[column]:.3f}" for column in headers[1:]]]
        for _, row in metric_frame.iterrows()
    ]
    table = table_axis.table(
        cellText=cells,
        colLabels=headers,
        cellLoc="center",
        colWidths=[0.20, *([0.133] * 6)],
        loc="center",
        bbox=[0.02, 0.08, 0.96, 0.75],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(13)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#d5dbe3")
        if row == 0:
            cell.set_facecolor("#202b3b")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_text_props(fontweight="bold")
    table_axis.set_title("LightGBM MAE / RMSE / R² Evolution", fontsize=20, fontweight="bold")

    fig.suptitle("Best Model Evolution: 26 → 34 → 35 Features", fontsize=25, fontweight="bold")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    metric_frame.to_csv(output_path.with_suffix(".csv"), index=False)
    return metric_frame
