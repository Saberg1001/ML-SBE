"""Compare 26-, 34-, and 35-feature models on experimental data."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "predictions" / "main" / "experimental_models_26_34_35_comparison.png"

MODEL_FILES = {
    "26 features": ROOT / "predictions" / "main" / "experimental-data_26_refactored_without_family" / "predictions.csv",
    "34 features": ROOT / "predictions" / "main" / "experimental-data_34_without_family" / "predictions.csv",
    "35 features": ROOT / "predictions" / "main" / "experimental-data_35_with_family" / "predictions.csv",
}

MODEL_COLORS = {
    "26 features": "#4C78A8",
    "34 features": "#F58518",
    "35 features": "#54A24B",
}

FAMILY_COLORS = {
    "argyrodites": "#4C78A8",
    "lgps": "#F58518",
    "halides": "#E45756",
    "thio_lisicon": "#72B7B2",
}


def load_predictions() -> dict[str, pd.DataFrame]:
    labels = pd.read_csv(ROOT / "rawdata" / "experimental-data-labeled.csv")
    label_columns = labels[["ID", "Family", "base_formula", "source_row"]]
    frames = {}
    for name, path in MODEL_FILES.items():
        frame = pd.read_csv(path)
        for column in ("Family", "base_formula", "source_row"):
            if column in frame.columns:
                frame = frame.drop(columns=column)
        frames[name] = frame.merge(label_columns, on="ID", how="left")
    return frames


def adjacent_accuracy(frame: pd.DataFrame) -> float:
    correct = 0
    comparable = 0
    ordered = frame.sort_values("source_row")
    for _, group in ordered.groupby("base_formula"):
        if len(group) < 2:
            continue
        true_delta = np.diff(group["true_log10_conductivity"].to_numpy())
        pred_delta = np.diff(group["pred_log10_conductivity"].to_numpy())
        mask = (true_delta != 0) & (pred_delta != 0)
        correct += int(np.sum(np.sign(true_delta[mask]) == np.sign(pred_delta[mask])))
        comparable += int(np.sum(mask))
    return correct / comparable


def metrics(frame: pd.DataFrame) -> dict[str, float]:
    true = frame["true_log10_conductivity"]
    pred = frame["pred_log10_conductivity"]
    return {
        "MAE": mean_absolute_error(true, pred),
        "RMSE": mean_squared_error(true, pred) ** 0.5,
        "Spearman": spearmanr(true, pred).statistic,
        "Adjacent accuracy": adjacent_accuracy(frame),
    }


def main() -> None:
    frames = load_predictions()
    all_values = np.concatenate(
        [
            np.concatenate(
                [
                    frame["true_log10_conductivity"].to_numpy(),
                    frame["pred_log10_conductivity"].to_numpy(),
                ]
            )
            for frame in frames.values()
        ]
    )
    lower = np.floor(all_values.min() * 2) / 2 - 0.1
    upper = np.ceil(all_values.max() * 2) / 2 + 0.1

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10})
    fig = plt.figure(figsize=(14.5, 8.4), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, height_ratios=[1.55, 1.0])

    for index, (name, frame) in enumerate(frames.items()):
        ax = fig.add_subplot(grid[0, index])
        for family, group in frame.groupby("Family"):
            ax.scatter(
                group["true_log10_conductivity"],
                group["pred_log10_conductivity"],
                s=30,
                alpha=0.78,
                color=FAMILY_COLORS.get(family, "#999999"),
                edgecolor="white",
                linewidth=0.35,
                label=family.replace("_", " "),
            )
        ax.plot([lower, upper], [lower, upper], "--", color="#333333", linewidth=1.2)
        model_metrics = metrics(frame)
        ax.text(
            0.04,
            0.96,
            f"MAE = {model_metrics['MAE']:.3f}\n"
            f"RMSE = {model_metrics['RMSE']:.3f}\n"
            f"Spearman = {model_metrics['Spearman']:.3f}",
            transform=ax.transAxes,
            va="top",
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.88, "edgecolor": "#BBBBBB"},
        )
        ax.set(xlim=(lower, upper), ylim=(lower, upper), aspect="equal")
        ax.set_title(name, fontweight="bold", color=MODEL_COLORS[name])
        ax.set_xlabel(r"Experimental log$_{10}$ conductivity (S cm$^{-1}$)")
        if index == 0:
            ax.set_ylabel(r"Predicted log$_{10}$ conductivity (S cm$^{-1}$)")
            ax.legend(frameon=False, fontsize=9, loc="lower right")
        ax.grid(alpha=0.18)

    summary = pd.DataFrame({name: metrics(frame) for name, frame in frames.items()}).T
    panels = [
        ("MAE", "MAE (log$_{10}$)", False),
        ("RMSE", "RMSE (log$_{10}$)", False),
        ("Spearman", "Spearman rank correlation", True),
    ]
    for index, (metric, ylabel, higher_is_better) in enumerate(panels):
        ax = fig.add_subplot(grid[1, index])
        names = list(frames)
        values = summary.loc[names, metric].to_numpy()
        bars = ax.bar(
            names,
            values,
            color=[MODEL_COLORS[name] for name in names],
            width=0.64,
            alpha=0.9,
        )
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}", ha="center", va="bottom")
        if metric == "Spearman":
            adjacent = summary.loc[names, "Adjacent accuracy"].to_numpy()
            twin = ax.twinx()
            twin.plot(names, adjacent, color="#7A5195", marker="D", linewidth=2, label="Adjacent accuracy")
            for x, value in zip(names, adjacent):
                twin.text(x, value + 0.018, f"{value:.0%}", color="#7A5195", ha="center", va="bottom")
            twin.set_ylim(0, 0.82)
            twin.set_ylabel("Adjacent trend accuracy", color="#7A5195")
            twin.tick_params(axis="y", colors="#7A5195")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{metric} ({'higher' if higher_is_better else 'lower'} is better)")
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", rotation=12)

    fig.suptitle("Experimental performance of 26-, 34-, and 35-feature models", fontsize=16, fontweight="bold")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
