from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
TEST_PREDICTIONS = (
    ROOT
    / "models"
    / "outputs_optuna"
    / "ionic_26_features_random_gt1e-6_50"
    / "lightgbm"
    / "test_predictions.csv"
)
EXPERIMENT_PREDICTIONS = ROOT / "predictions" / "expriment-test_predictions.csv"
OUTPUT_DIR = ROOT / "predictions"
FONT_SIZE = 10
TEST_COLOR = "#1f77b4"
EXPERIMENT_COLOR = "#d62728"


def read_test_residuals() -> pd.DataFrame:
    frame = pd.read_csv(TEST_PREDICTIONS)
    output = frame[["ID", "y_true", "y_pred"]].copy()
    output = output.rename(columns={"y_true": "true_log10", "y_pred": "pred_log10"})
    output["residual_pred_minus_true"] = output["pred_log10"] - output["true_log10"]
    output["dataset"] = "Original test set"
    return output.dropna(subset=["residual_pred_minus_true"])


def read_experiment_residuals() -> pd.DataFrame:
    frame = pd.read_csv(EXPERIMENT_PREDICTIONS)
    output = frame[
        ["ID", "True Composition", "true_log10_conductivity", "pred_log10_conductivity", "status"]
    ].copy()
    output = output[output["status"].eq("ok")]
    output = output.rename(
        columns={
            "true_log10_conductivity": "true_log10",
            "pred_log10_conductivity": "pred_log10",
        }
    )
    output["residual_pred_minus_true"] = output["pred_log10"] - output["true_log10"]
    output["dataset"] = "New experimental data"
    return output.dropna(subset=["residual_pred_minus_true"])


def summary_row(name: str, residuals: pd.Series) -> dict[str, float | int | str]:
    values = residuals.to_numpy()
    return {
        "dataset": name,
        "n": len(values),
        "mean_residual": float(np.mean(values)),
        "median_residual": float(np.median(values)),
        "std_residual": float(np.std(values, ddof=1)),
        "mae": float(np.mean(np.abs(values))),
        "rmse": float(np.sqrt(np.mean(values**2))),
        "q05": float(np.quantile(values, 0.05)),
        "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)),
        "q95": float(np.quantile(values, 0.95)),
        "underpredicted_fraction": float(np.mean(values < 0)),
        "within_1_log_unit_fraction": float(np.mean(np.abs(values) <= 1.0)),
    }


def save_summary(test: pd.DataFrame, experiment: pd.DataFrame) -> Path:
    rows = [
        summary_row("Original test set", test["residual_pred_minus_true"]),
        summary_row("New experimental data", experiment["residual_pred_minus_true"]),
    ]
    output = OUTPUT_DIR / "residual_test_vs_experiment_summary.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    return output


def main() -> None:
    test = read_test_residuals()
    experiment = read_experiment_residuals()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = save_summary(test, experiment)

    test_residuals = test["residual_pred_minus_true"].to_numpy()
    exp_residuals = experiment["residual_pred_minus_true"].to_numpy()
    all_residuals = np.concatenate([test_residuals, exp_residuals])
    lower = np.floor(all_residuals.min() - 0.25)
    upper = np.ceil(all_residuals.max() + 0.25)

    plt.rcParams.update(
        {
            "font.size": FONT_SIZE,
            "axes.titlesize": FONT_SIZE,
            "axes.labelsize": FONT_SIZE,
            "xtick.labelsize": FONT_SIZE,
            "ytick.labelsize": FONT_SIZE,
            "legend.fontsize": FONT_SIZE,
        }
    )
    fig, ax_box = plt.subplots(figsize=(4.8, 5.2), dpi=220, constrained_layout=True)

    box = ax_box.boxplot(
        [test_residuals, exp_residuals],
        vert=True,
        patch_artist=True,
        widths=0.58,
        showfliers=False,
        tick_labels=["Test", "Experiment"],
    )
    for patch, color in zip(box["boxes"], [TEST_COLOR, EXPERIMENT_COLOR]):
        patch.set_facecolor(color)
        patch.set_alpha(0.28)
        patch.set_edgecolor(color)
    for element in ["whiskers", "caps", "medians"]:
        for item in box[element]:
            item.set_color("0.25")

    rng = np.random.default_rng(42)
    ax_box.scatter(
        rng.normal(1, 0.045, len(test_residuals)),
        test_residuals,
        s=16,
        color=TEST_COLOR,
        alpha=0.55,
        edgecolors="none",
    )
    ax_box.scatter(
        rng.normal(2, 0.045, len(exp_residuals)),
        exp_residuals,
        s=16,
        color=EXPERIMENT_COLOR,
        alpha=0.55,
        edgecolors="none",
    )
    ax_box.axhline(0, color="black", linewidth=1.1)
    ax_box.axhline(-1, color="0.65", linestyle="--", linewidth=0.9)
    ax_box.axhline(1, color="0.65", linestyle="--", linewidth=0.9)
    ax_box.set_ylabel("Residual: predicted - true log10 conductivity")
    ax_box.set_title("Residual Spread")
    ax_box.grid(True, axis="y", color="0.9", linewidth=0.8)
    ax_box.set_ylim(lower, upper)

    png_path = OUTPUT_DIR / "residual_test_vs_experiment_log10.png"
    pdf_path = OUTPUT_DIR / "residual_test_vs_experiment_log10.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
