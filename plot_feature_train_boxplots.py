from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "features" / "ionic_26_features_train.csv"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "feature_train_boxplots"
NON_NUMERIC_COLUMNS = {"ID", "True Composition", r"$Z_{\mathrm{by}\ \mathrm{element}}$"}
CONDUCTIVITY_COLUMN = "Ionic conductivity (S cm-1)"
CONDUCTIVITY_LOG_COLUMN = r"$\log_{10}(\sigma)\ (\mathrm{S}\ \mathrm{cm}^{-1})$"
CONDUCTIVITY_BELOW_LIMIT = "1E-10"
CONDUCTIVITY_BELOW_LIMIT_VALUE = 1e-13


def configure_matplotlib(output_dir: Path):
    cache_dir = output_dir / "matplotlib_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def coerce_numeric(series: pd.Series, column: str) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        values = pd.to_numeric(series, errors="coerce")
    else:
        cleaned = (
            series.astype("string")
            .str.strip()
            .str.replace(",", "", regex=False)
        )
        if column == CONDUCTIVITY_COLUMN:
            below_limit = cleaned.str.upper().str.replace(" ", "", regex=False) == f"<{CONDUCTIVITY_BELOW_LIMIT}"
            cleaned = cleaned.mask(below_limit, str(CONDUCTIVITY_BELOW_LIMIT_VALUE))
        cleaned = cleaned.str.replace(r"^[<>]\s*", "", regex=True)
        values = pd.to_numeric(cleaned, errors="coerce")
    if column != CONDUCTIVITY_COLUMN:
        return values
    positive = values.where(values > 0)
    return np.log10(positive)


def output_column_name(column: str) -> str:
    if column == CONDUCTIVITY_COLUMN:
        return CONDUCTIVITY_LOG_COLUMN
    return column


def numeric_data(frame: pd.DataFrame) -> pd.DataFrame:
    numeric_columns: dict[str, pd.Series] = {}
    for column in frame.columns:
        if column in NON_NUMERIC_COLUMNS:
            continue
        values = coerce_numeric(frame[column], column)
        if values.notna().any():
            numeric_columns[output_column_name(column)] = values
    return pd.DataFrame(numeric_columns)


def iqr_summary(values: pd.Series) -> dict[str, float | int]:
    valid = values.dropna()
    count = int(valid.size)
    missing = int(values.isna().sum())
    if count == 0:
        return {
            "count": 0,
            "missing_count": missing,
            "missing_ratio": 1.0,
            "min": math.nan,
            "q1": math.nan,
            "median": math.nan,
            "q3": math.nan,
            "max": math.nan,
            "iqr": math.nan,
            "lower_fence": math.nan,
            "upper_fence": math.nan,
            "outlier_count": 0,
            "outlier_ratio": math.nan,
        }
    q1 = float(valid.quantile(0.25))
    median = float(valid.quantile(0.50))
    q3 = float(valid.quantile(0.75))
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    outliers = valid[(valid < lower_fence) | (valid > upper_fence)]
    return {
        "count": count,
        "missing_count": missing,
        "missing_ratio": missing / len(values),
        "min": float(valid.min()),
        "q1": q1,
        "median": median,
        "q3": q3,
        "max": float(valid.max()),
        "iqr": iqr,
        "lower_fence": lower_fence,
        "upper_fence": upper_fence,
        "outlier_count": int(outliers.size),
        "outlier_ratio": outliers.size / count,
    }


def write_summary(data: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    rows = []
    for column in data.columns:
        rows.append({"feature": column, **iqr_summary(data[column])})
    summary = pd.DataFrame(rows)
    summary.sort_values(
        by=["outlier_ratio", "missing_ratio"],
        ascending=[False, False],
        inplace=True,
    )
    summary.to_csv(output_path, index=False, encoding="utf-8")
    return summary


def should_use_log_scale(values: pd.Series) -> bool:
    valid = values.dropna()
    positive = valid[valid > 0]
    if positive.size != valid.size or positive.empty:
        return False
    return float(positive.max()) / float(positive.min()) >= 1000


def plot_raw_boxplots(data: pd.DataFrame, output_path: Path, plt) -> None:
    columns = list(data.columns)
    ncols = 3
    nrows = math.ceil(len(columns) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(27, max(8, nrows * 5.2)))
    axes = np.asarray(axes).reshape(-1)

    for ax, column in zip(axes, columns):
        values = data[column].dropna()
        ax.boxplot(
            values,
            vert=True,
            showmeans=True,
            meanline=True,
            patch_artist=True,
            boxprops={"facecolor": "#8ecae6", "edgecolor": "#1f4e79", "linewidth": 1.8},
            whiskerprops={"color": "#1f4e79", "linewidth": 1.6},
            capprops={"color": "#1f4e79", "linewidth": 1.6},
            medianprops={"color": "#d62828", "linewidth": 2.2},
            meanprops={"color": "#2a9d8f", "linewidth": 2.2},
            flierprops={
                "marker": "o",
                "markerfacecolor": "#f4a261",
                "markeredgecolor": "#9c4221",
                "markersize": 5,
                "alpha": 0.75,
            },
        )
        ax.set_title(column, fontsize=19)
        ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        ax.tick_params(axis="y", labelsize=17)
        ax.grid(axis="y", linestyle=":", linewidth=1.0, alpha=0.7)
        if should_use_log_scale(values):
            ax.set_yscale("log")
            ax.set_ylabel("log scale", fontsize=17)
        else:
            ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))

    for ax in axes[len(columns):]:
        ax.axis("off")

    fig.suptitle("feature_train numeric columns: raw boxplots", fontsize=24)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def robust_scaled_data(data: pd.DataFrame) -> pd.DataFrame:
    scaled: dict[str, pd.Series] = {}
    for column in data.columns:
        values = data[column]
        median = values.median(skipna=True)
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        scale = q3 - q1
        if pd.isna(scale) or scale == 0:
            scale = values.std(skipna=True)
        if pd.isna(scale) or scale == 0:
            continue
        scaled[column] = (values - median) / scale
    return pd.DataFrame(scaled)


def plot_robust_boxplot(data: pd.DataFrame, output_path: Path, plt) -> None:
    scaled = robust_scaled_data(data)
    columns = list(scaled.columns)
    values = [scaled[column].dropna() for column in columns]

    fig_height = max(16, len(columns) * 0.74)
    fig, ax = plt.subplots(figsize=(25, fig_height))
    ax.boxplot(
        values,
        vert=False,
        showfliers=True,
        showmeans=True,
        meanline=True,
        patch_artist=True,
        boxprops={"facecolor": "#b7e4c7", "edgecolor": "#1b4332", "linewidth": 1.9},
        whiskerprops={"color": "#1b4332", "linewidth": 1.7},
        capprops={"color": "#1b4332", "linewidth": 1.7},
        medianprops={"color": "#d00000", "linewidth": 2.4},
        meanprops={"color": "#0077b6", "linewidth": 2.4},
        flierprops={
            "marker": "o",
            "markerfacecolor": "#ffb703",
            "markeredgecolor": "#99582a",
            "markersize": 5.5,
            "alpha": 0.7,
        },
    )
    ax.set_yticks(range(1, len(columns) + 1), columns, fontsize=18)
    ax.set_xscale("symlog", linthresh=5)
    ax.axvline(0, color="#343a40", linewidth=1.4, alpha=0.8)
    ax.grid(axis="x", which="both", linestyle=":", linewidth=1.0, alpha=0.65)
    ax.tick_params(axis="x", labelsize=18)
    ax.set_xlabel("Robust-scaled value: (x - median) / IQR, symlog x-axis", fontsize=20)
    ax.set_title("feature_train numeric columns: robust-scaled boxplots", fontsize=24)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate feature_train.csv numeric data with boxplots and IQR outlier summary."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt = configure_matplotlib(args.output_dir)

    frame = pd.read_csv(args.input)
    data = numeric_data(frame)
    if data.empty:
        raise RuntimeError(f"No numeric columns found in {args.input}")

    summary_path = args.output_dir / "feature_train_boxplot_summary.csv"
    raw_plot_path = args.output_dir / "feature_train_boxplots_raw.png"
    robust_plot_path = args.output_dir / "feature_train_boxplots_robust_scaled.png"

    summary = write_summary(data, summary_path)
    plot_raw_boxplots(data, raw_plot_path, plt)
    plot_robust_boxplot(data, robust_plot_path, plt)

    print(f"Rows: {len(frame)}")
    print(f"Numeric columns evaluated: {len(data.columns)}")
    print(f"Summary: {summary_path}")
    print(f"Raw boxplots: {raw_plot_path}")
    print(f"Robust-scaled boxplots: {robust_plot_path}")
    print("Highest outlier ratios:")
    print(summary[["feature", "outlier_count", "outlier_ratio", "missing_ratio"]].head(8).to_string(index=False))


if __name__ == "__main__":
    main()
