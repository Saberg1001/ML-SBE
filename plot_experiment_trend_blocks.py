from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


INPUT_CSV = Path("predictions/experimental-data/predictions.csv")
OUTPUT_PNG = Path("predictions/true_black_pred_blue_trend_blocks_formula_legend.png")


@dataclass(frozen=True)
class SeriesBlock:
    start_id: int
    end_id: int
    label: str


SERIES_BLOCKS = [
    SeriesBlock(1, 4, "Li(5.7-x)PS(4.7-x)Cl(1.3+x)"),
    SeriesBlock(5, 9, "Li5.5PS4.5Cl(1.5-x)Ix"),
    SeriesBlock(10, 16, "Li5.4PS(4.4-x)OxClBr0.6"),
    SeriesBlock(17, 21, "Li5.5AsS(4.5-x)OxBr1.5"),
    SeriesBlock(22, 24, "Li5.5As/SbS(4.5-x)OxBr1.5"),
    SeriesBlock(25, 28, "Li5.4PS4O0.4Cl(1-x)FxBr0.6"),
    SeriesBlock(29, 32, "Li(6.3+x)As(0.7-x)SixS5I"),
    SeriesBlock(33, 37, "Li(7.0-x)SbxSi(1-x)S5I"),
    SeriesBlock(38, 42, "Li(7.7-x)Sb0.3Si0.7S(6-x)I(x)"),
    SeriesBlock(43, 44, "Li3-3.875 As/Sn sulfides"),
    SeriesBlock(45, 48, "Li5.4P(1-x)SnxS4.4ClBr0.6"),
    SeriesBlock(49, 52, "Li5.4P/Sn/Si S(4.4-x)OxClBr0.6"),
    SeriesBlock(53, 56, "Li10SnP/Sb S/O series"),
    SeriesBlock(57, 61, "Li10 Si/Sn P2 S(12-x)Ox"),
    SeriesBlock(62, 67, "Li6.7Si0.7Sb0.3S(5-x)Se/OxI"),
    SeriesBlock(68, 70, "Li6.7Si0.7Sb(0.3-x)AsxS4.8O0.2I"),
    SeriesBlock(71, 72, "Li6.7Si0.7Sb0.3S/Se/O/I"),
    SeriesBlock(73, 79, "Li-Zr-O-Cl mixed series"),
    SeriesBlock(80, 85, "Li(5.4+x)P(1-x)ZrxS(4.4-2x)O(2x)ClBr0.6"),
    SeriesBlock(86, 86, "Li3.45Sn0.09Si0.36P0.55S3.65O0.35"),
    SeriesBlock(87, 90, "Li9.54Si1.566Sn0.174P1.34SbxS11.1Br0.3O0.6"),
    SeriesBlock(91, 93, "Li9.54Si1.566Sn0.174P1.34AsxS11.1Br0.3O0.6"),
    SeriesBlock(94, 97, "Li9.54Si1.74P1.44S(11.7-x)Br0.3Ox"),
    SeriesBlock(98, 107, "Li9.54(Si(1-x)Gex)1.74P1.44S11.1Br0.3O0.6"),
    SeriesBlock(108, 113, "Li9.54(Si(1-x)Snx)1.74P1.44S11.1Br0.3O0.6"),
]


COLORS = {
    "all": "#dff0df",
    "mixed": "#fff3bf",
    "opposite": "#f5dddd",
    "single": "#eeeeee",
}


def experiment_number(series: pd.Series) -> pd.Series:
    return series.str.extract(r"(\d+)", expand=False).astype(int)


def trend_match_count(frame: pd.DataFrame) -> tuple[int, int]:
    true_delta = np.diff(frame["true_log10_conductivity"].to_numpy())
    pred_delta = np.diff(frame["pred_log10_conductivity"].to_numpy())
    count = int((true_delta * pred_delta > 0).sum())
    return count, len(true_delta)


def block_color(matches: int, comparisons: int) -> str:
    if comparisons == 0:
        return COLORS["single"]
    if matches == comparisons:
        return COLORS["all"]
    if matches >= ceil(comparisons / 2):
        return COLORS["mixed"]
    return COLORS["opposite"]


def main() -> None:
    data = pd.read_csv(INPUT_CSV)
    data = data[data["status"].eq("ok")].copy()
    data["exp_no"] = experiment_number(data["ID"])
    data = data.sort_values("exp_no").reset_index(drop=True)
    data["x"] = np.arange(len(data))

    fig = plt.figure(figsize=(28, 12), dpi=220)
    grid = fig.add_gridspec(1, 2, width_ratios=[4.9, 1.7], wspace=0.04)
    ax = fig.add_subplot(grid[0, 0])
    ax_info = fig.add_subplot(grid[0, 1])

    y_min = min(data["true_log10_conductivity"].min(), data["pred_log10_conductivity"].min()) - 0.22
    y_max = max(data["true_log10_conductivity"].max(), data["pred_log10_conductivity"].max()) + 0.28

    summary_lines = []
    for index, block in enumerate(SERIES_BLOCKS, start=1):
        mask = data["exp_no"].between(block.start_id, block.end_id)
        segment = data[mask]
        if segment.empty:
            continue

        left = segment["x"].min() - 0.5
        right = segment["x"].max() + 0.5
        matches, comparisons = trend_match_count(segment)
        match_text = f"{matches}/{comparisons}" if comparisons else "n/a"
        ax.axvspan(left, right, color=block_color(matches, comparisons), ec="#777777", lw=1.0, alpha=0.82)
        ax.axvline(right, color="#777777", lw=0.9, alpha=0.72)

        center = (left + right) / 2
        ax.text(center, y_max - 0.06, str(index), ha="center", va="top", fontsize=10, color="#333333")
        ax.text(center, y_max - 0.18, match_text, ha="center", va="top", fontsize=9, color="#333333")

        x_values = segment["x"].to_numpy()
        ax.plot(
            x_values,
            segment["true_log10_conductivity"],
            color="black",
            marker="o",
            lw=2.3,
            ms=5.0,
            zorder=3,
        )
        ax.plot(
            x_values,
            segment["pred_log10_conductivity"],
            color="#1f77b4",
            marker="s",
            linestyle=(0, (6, 4)),
            lw=2.5,
            ms=4.8,
            zorder=3,
        )
        summary_lines.append(f"{index}. {block.label}  ({match_text})")

    ax.set_xlim(-0.5, len(data) - 0.5)
    ax.set_ylim(y_min, y_max)
    ax.set_title("True vs Predicted Conductivity Trend by Composition Series", fontsize=18, pad=9)
    ax.set_ylabel("log10 conductivity", fontsize=13)
    ax.set_xlabel("sample order in CSV", fontsize=13)
    tick_positions = np.arange(0, len(data), 5)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(data.loc[tick_positions, "ID"], rotation=60, ha="right", fontsize=9)
    ax.tick_params(axis="y", labelsize=11)
    ax.grid(axis="y", color="#b0b0b0", alpha=0.28, lw=0.9)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

    ax_info.axis("off")
    handles = [
        Line2D([0], [0], color="black", marker="o", lw=2.3, ms=6, label="True log10 conductivity"),
        Line2D(
            [0],
            [0],
            color="#1f77b4",
            marker="s",
            linestyle=(0, (6, 4)),
            lw=2.5,
            ms=6,
            label="Predicted log10 conductivity",
        ),
        Patch(facecolor=COLORS["all"], edgecolor="#777777", label="All adjacent directions match"),
        Patch(facecolor=COLORS["mixed"], edgecolor="#777777", label="Mixed: at least half directions match"),
        Patch(facecolor=COLORS["opposite"], edgecolor="#777777", label="Mostly opposite"),
        Patch(facecolor=COLORS["single"], edgecolor="#777777", label="Single point"),
    ]
    ax_info.legend(
        handles=handles,
        loc="upper left",
        title="Legend",
        frameon=True,
        fontsize=10.5,
        title_fontsize=13,
        borderpad=1.1,
        labelspacing=1.0,
        handlelength=2.2,
    )

    wrapped_lines = []
    for line in summary_lines:
        wrapped = textwrap.wrap(line, width=58, subsequent_indent="    ")
        wrapped_lines.extend(wrapped)
    ax_info.text(
        0.0,
        0.48,
        "\n".join(wrapped_lines),
        transform=ax_info.transAxes,
        ha="left",
        va="top",
        fontsize=8.4,
        family="DejaVu Sans",
        bbox={"facecolor": "white", "edgecolor": "#cccccc", "boxstyle": "square,pad=0.45"},
    )

    fig.savefig(OUTPUT_PNG, facecolor="white", bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


if __name__ == "__main__":
    main()
