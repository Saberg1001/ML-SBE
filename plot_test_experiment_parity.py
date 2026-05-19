from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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


def metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": mean_squared_error(y_true, y_pred) ** 0.5,
        "r2": r2_score(y_true, y_pred) if len(y_true) > 1 else np.nan,
    }


def read_test_predictions() -> pd.DataFrame:
    frame = pd.read_csv(TEST_PREDICTIONS)
    frame = frame.rename(columns={"y_true": "true_log10", "y_pred": "pred_log10"})
    frame = frame[["ID", "true_log10", "pred_log10"]].copy()
    frame["residual_pred_minus_true"] = frame["pred_log10"] - frame["true_log10"]
    frame["dataset"] = "Original test set"
    return frame.dropna(subset=["true_log10", "pred_log10"])


def read_experiment_predictions() -> pd.DataFrame:
    frame = pd.read_csv(EXPERIMENT_PREDICTIONS)
    frame = frame.rename(
        columns={
            "true_log10_conductivity": "true_log10",
            "pred_log10_conductivity": "pred_log10",
        }
    )
    frame = frame[["ID", "True Composition", "true_log10", "pred_log10", "status"]].copy()
    frame = frame[frame["status"].eq("ok")]
    frame["residual_pred_minus_true"] = frame["pred_log10"] - frame["true_log10"]
    frame["dataset"] = "New experimental data"
    return frame.dropna(subset=["true_log10", "pred_log10"])


def save_metrics(test: pd.DataFrame, experiment: pd.DataFrame) -> Path:
    rows = []
    for name, frame in [("Original test set", test), ("New experimental data", experiment)]:
        values = metrics(frame["true_log10"], frame["pred_log10"])
        rows.append({"dataset": name, "n": len(frame), **values})
    output = OUTPUT_DIR / "parity_test_vs_experiment_metrics.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    return output


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


def save_residual_summary(test: pd.DataFrame, experiment: pd.DataFrame) -> Path:
    rows = [
        summary_row("Original test set", test["residual_pred_minus_true"]),
        summary_row("New experimental data", experiment["residual_pred_minus_true"]),
    ]
    output = OUTPUT_DIR / "residual_test_vs_experiment_summary.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    return output


def plot_parity(ax: plt.Axes, test: pd.DataFrame, experiment: pd.DataFrame) -> None:
    all_values = pd.concat(
        [
            test[["true_log10", "pred_log10"]],
            experiment[["true_log10", "pred_log10"]],
        ],
        ignore_index=True,
    )
    lower = np.floor(all_values.min().min() - 0.25)
    upper = np.ceil(all_values.max().max() + 0.25)

    for frame, color, marker, label in [
        (test, TEST_COLOR, "o", "Original test set"),
        (experiment, EXPERIMENT_COLOR, "^", "New experimental data"),
    ]:
        values = metrics(frame["true_log10"], frame["pred_log10"])
        ax.scatter(
            frame["true_log10"],
            frame["pred_log10"],
            s=34,
            c=color,
            marker=marker,
            alpha=0.78,
            edgecolors="white",
            linewidths=0.4,
            label=(
                f"{label} (MAE={values['mae']:.2f}, RMSE={values['rmse']:.2f})"
            ),
        )

    ax.plot([lower, upper], [lower, upper], color="black", linewidth=1.2, label="Parity")
    ax.plot([lower, upper], [lower + 1, upper + 1], color="0.65", linestyle="--", linewidth=0.9)
    ax.plot([lower, upper], [lower - 1, upper - 1], color="0.65", linestyle="--", linewidth=0.9)
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_aspect("equal", adjustable="box")
    ax.set_box_aspect(1)
    ax.set_xlabel("True log10 ionic conductivity (S cm-1)")
    ax.set_ylabel("Predicted log10 ionic conductivity (S cm-1)")
    ax.set_title("Parity Plot")
    ax.grid(True, color="0.9", linewidth=0.8)
    ax.legend(frameon=False, loc="upper left")


def plot_residual_spread(ax: plt.Axes, test: pd.DataFrame, experiment: pd.DataFrame) -> None:
    test_residuals = test["residual_pred_minus_true"].to_numpy()
    exp_residuals = experiment["residual_pred_minus_true"].to_numpy()
    all_residuals = np.concatenate([test_residuals, exp_residuals])
    lower = np.floor(all_residuals.min() - 0.25)
    upper = np.ceil(all_residuals.max() + 0.25)

    box = ax.boxplot(
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
    ax.scatter(
        rng.normal(1, 0.045, len(test_residuals)),
        test_residuals,
        s=15,
        color=TEST_COLOR,
        alpha=0.55,
        edgecolors="none",
    )
    ax.scatter(
        rng.normal(2, 0.045, len(exp_residuals)),
        exp_residuals,
        s=15,
        color=EXPERIMENT_COLOR,
        alpha=0.55,
        edgecolors="none",
    )
    ax.axhline(0, color="black", linewidth=1.1)
    ax.axhline(-1, color="0.65", linestyle="--", linewidth=0.9)
    ax.axhline(1, color="0.65", linestyle="--", linewidth=0.9)
    ax.set_ylabel("Residual: predicted - true log10 conductivity")
    ax.set_title("Residual Spread")
    ax.grid(True, axis="y", color="0.9", linewidth=0.8)
    ax.set_ylim(lower, upper)
    ax.set_box_aspect(1)


def main() -> None:
    test = read_test_predictions()
    experiment = read_experiment_predictions()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = save_metrics(test, experiment)
    summary_path = save_residual_summary(test, experiment)

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
    fig, (ax_parity, ax_residual) = plt.subplots(
        1,
        2,
        figsize=(10.8, 5.4),
        dpi=220,
        gridspec_kw={"width_ratios": [1.0, 1.0]},
        constrained_layout=True,
    )
    plot_parity(ax_parity, test, experiment)
    plot_residual_spread(ax_residual, test, experiment)

    png_path = OUTPUT_DIR / "parity_test_vs_experiment_log10.png"
    pdf_path = OUTPUT_DIR / "parity_test_vs_experiment_log10.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")
    print(f"Wrote {metrics_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
