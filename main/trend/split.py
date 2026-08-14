"""Create a deterministic DOI-level train/validation split."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Allow running this module directly as a script (VS Code "Run" button): set
# __package__ so the relative imports below resolve, and expose project root.
if __package__ is None:
    _FILE = Path(__file__).resolve()
    if str(_FILE.parents[2]) not in sys.path:
        sys.path.insert(0, str(_FILE.parents[2]))
    __package__ = f"{_FILE.parents[1].name}.{_FILE.parents[0].name}"

import hashlib
import json

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

from ..paths import TREND_DIR, portable_path
from .features import MODEL_FEATURE_COLUMNS


DEFAULT_INPUT = TREND_DIR / "data-trend-v2-pairs-feature.csv"
DEFAULT_TRAIN = TREND_DIR / "data-trend-v2-pairs-feature-train.csv"
DEFAULT_VALIDATION = TREND_DIR / "data-trend-v2-pairs-feature-validation.csv"
DEFAULT_ASSIGNMENTS = TREND_DIR / "data-trend-v2-split-assignments.csv"
DEFAULT_MANIFEST = TREND_DIR / "data-trend-v2-split-manifest.json"
TARGET_VALIDATION_FRACTION = 0.10
SPLIT_VERSION = "trend_doi_stratified_split_v3"

# Only model inputs, target, weight, and identifiers required for grouped
# training/evaluation are written to the split files.
OUTPUT_COLUMNS = [
    "pair_id",
    "group_id",
    "doi",
    "化学式_a",
    "化学式_b",
    "family",
    "trend_label",
    "pair_weight_group_equal",
    *MODEL_FEATURE_COLUMNS,
]


def _stable_key(doi: str) -> tuple[str, str]:
    digest = hashlib.sha256(f"{SPLIT_VERSION}|{doi}".encode()).hexdigest()
    return digest, doi


def _target_label_counts(
    label_counts: pd.Series,
    target_rows: int,
) -> dict[str, int]:
    """Allocate integer validation targets with the largest-remainder rule."""
    exact = label_counts.astype(float) * target_rows / int(label_counts.sum())
    target = np.floor(exact).astype(int)
    remaining = target_rows - int(target.sum())
    order = sorted(
        label_counts.index,
        key=lambda label: (-(exact[label] - target[label]), str(label)),
    )
    for label in order[:remaining]:
        target[label] += 1
    return {str(label): int(target[label]) for label in label_counts.index}


def _choose_dois(
    doi_label_counts: pd.DataFrame,
    target_rows: int,
    target_label_counts: dict[str, int],
) -> set[str]:
    """Optimize validation size and class balance under indivisible DOI groups."""
    labels = list(target_label_counts)
    table = doi_label_counts.reindex(columns=labels, fill_value=0).astype(int)
    dois = list(table.index.astype(str))
    if len(dois) < 2:
        raise ValueError("At least two DOI groups are required for splitting.")
    for label in labels:
        if int((table[label] > 0).sum()) < 2:
            raise ValueError(
                f"Label {label!r} must occur in at least two DOI groups."
            )

    # Binary DOI decisions plus positive/negative absolute-deviation variables.
    deviation_count = 1 + len(labels)
    variable_count = len(dois) + 2 * deviation_count
    objective = np.zeros(variable_count, dtype=float)
    total_rows = int(table.to_numpy().sum())
    row_priority = float(2 * total_rows + 1)
    objective[len(dois):len(dois) + 2] = row_priority
    objective[len(dois) + 2:] = 1.0
    rank = {
        doi: index + 1
        for index, doi in enumerate(sorted(dois, key=_stable_key))
    }
    tie_scale = 1.0 / (len(dois) + 1) ** 3
    objective[:len(dois)] = [rank[doi] * tie_scale for doi in dois]

    equations = np.zeros((deviation_count, variable_count), dtype=float)
    targets = np.array(
        [target_rows, *[target_label_counts[label] for label in labels]],
        dtype=float,
    )
    equations[0, :len(dois)] = table.sum(axis=1).to_numpy(dtype=float)
    for index, label in enumerate(labels, start=1):
        equations[index, :len(dois)] = table[label].to_numpy(dtype=float)
    for index in range(deviation_count):
        equations[index, len(dois) + 2 * index] = -1.0
        equations[index, len(dois) + 2 * index + 1] = 1.0

    coverage = np.zeros((1 + len(labels), variable_count), dtype=float)
    coverage[0, :len(dois)] = 1.0
    lower = [1.0]
    upper = [float(len(dois) - 1)]
    for index, label in enumerate(labels, start=1):
        coverage[index, :len(dois)] = table[label].to_numpy(dtype=float)
        lower.append(1.0)
        upper.append(float(table[label].sum() - 1))

    result = milp(
        c=objective,
        integrality=np.r_[np.ones(len(dois)), np.zeros(2 * deviation_count)],
        bounds=Bounds(
            np.zeros(variable_count),
            np.r_[np.ones(len(dois)), np.full(2 * deviation_count, np.inf)],
        ),
        constraints=[
            LinearConstraint(equations, targets, targets),
            LinearConstraint(coverage, np.array(lower), np.array(upper)),
        ],
        options={"mip_rel_gap": 0.0},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"DOI split optimization failed: {result.message}")
    return {doi for doi, selected in zip(dois, result.x[:len(dois)]) if selected > 0.5}


def split_feature_file(
    input_path: str | Path = DEFAULT_INPUT,
    train_path: str | Path = DEFAULT_TRAIN,
    validation_path: str | Path = DEFAULT_VALIDATION,
    assignments_path: str | Path = DEFAULT_ASSIGNMENTS,
    manifest_path: str | Path = DEFAULT_MANIFEST,
    *,
    target_fraction: float = TARGET_VALIDATION_FRACTION,
) -> dict[str, object]:
    input_path = Path(input_path)
    frame = pd.read_csv(input_path, dtype=str, keep_default_na=False)
    missing = sorted(set(OUTPUT_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Input feature table is missing columns: {missing}")
    if frame["pair_id"].duplicated().any():
        raise ValueError("pair_id must be unique before splitting.")
    if not 0 < target_fraction < 1:
        raise ValueError("target_fraction must be between 0 and 1.")
    doi_label_counts = pd.crosstab(frame["doi"], frame["trend_label"])
    label_counts = frame["trend_label"].value_counts().sort_index()
    target_rows = round(len(frame) * target_fraction)
    target_labels = _target_label_counts(label_counts, target_rows)
    validation_dois = _choose_dois(
        doi_label_counts,
        target_rows,
        target_labels,
    )
    validation_mask = frame["doi"].isin(validation_dois)
    train_full = frame.loc[~validation_mask]
    validation_full = frame.loc[validation_mask]
    if set(train_full["doi"]) & set(validation_full["doi"]):
        raise ValueError("DOI leakage detected.")
    labels = set(frame["trend_label"])
    if not labels <= set(train_full["trend_label"]):
        raise ValueError("Training split is missing a trend label.")
    if not labels <= set(validation_full["trend_label"]):
        raise ValueError("Validation split is missing a trend label.")
    train = train_full.loc[:, OUTPUT_COLUMNS].copy()
    validation = validation_full.loc[:, OUTPUT_COLUMNS].copy()
    train_path, validation_path = Path(train_path), Path(validation_path)
    assignments_path, manifest_path = Path(assignments_path), Path(manifest_path)
    for path in (train_path, validation_path, assignments_path, manifest_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    train.to_csv(train_path, index=False)
    validation.to_csv(validation_path, index=False)
    assignments = pd.DataFrame([
        {
            "doi": doi,
            "split": "validation" if doi in validation_dois else "train",
            "pair_rows": int(doi_label_counts.loc[doi].sum()),
            **{
                f"{label}_rows": int(doi_label_counts.loc[doi].get(label, 0))
                for label in label_counts.index
            },
            "selection_key": _stable_key(doi)[0],
        }
        for doi in sorted(doi_label_counts.index)
    ])
    assignments.to_csv(assignments_path, index=False)
    manifest = {
        "schema_version": SPLIT_VERSION,
        "input": portable_path(input_path),
        "target_validation_fraction": target_fraction,
        "actual_validation_fraction": len(validation) / len(frame),
        "selection_method": (
            "mixed-integer optimization: row-count deviation first, "
            "then label-count deviation; DOI groups are indivisible"
        ),
        "output_columns": OUTPUT_COLUMNS,
        "counts": {
            "input_rows": len(frame),
            "train_rows": len(train),
            "validation_rows": len(validation),
            "train_dois": train["doi"].nunique(),
            "validation_dois": validation["doi"].nunique(),
        },
        "label_distribution": {
            "overall": frame["trend_label"].value_counts().to_dict(),
            "train": train["trend_label"].value_counts().to_dict(),
            "validation": validation["trend_label"].value_counts().to_dict(),
            "validation_target": target_labels,
        },
        "label_fraction": {
            "overall": frame["trend_label"].value_counts(normalize=True).to_dict(),
            "train": train["trend_label"].value_counts(normalize=True).to_dict(),
            "validation": validation["trend_label"].value_counts(normalize=True).to_dict(),
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    """Split the trend feature table into train/validation (point-run entry).

    Run directly via ``python main/trend/split.py`` (the "Run" button).
    Input    : data/trend/data-trend-v2-pairs-feature.csv
    Outputs  : data/trend/data-trend-v2-pairs-feature-train.csv
               data/trend/data-trend-v2-pairs-feature-validation.csv
               data/trend/data-trend-v2-split-assignments.csv
               data/trend/data-trend-v2-split-manifest.json
    """
    result = split_feature_file(
        DEFAULT_INPUT,
        DEFAULT_TRAIN,
        DEFAULT_VALIDATION,
        DEFAULT_ASSIGNMENTS,
        DEFAULT_MANIFEST,
        target_fraction=TARGET_VALIDATION_FRACTION,
    )
    counts = result["counts"]
    print(f"input_rows={counts['input_rows']} "
          f"train_rows={counts['train_rows']} validation_rows={counts['validation_rows']}")
    print(f"Train CSV : {DEFAULT_TRAIN.resolve()}")
    print(f"Val CSV   : {DEFAULT_VALIDATION.resolve()}")


if __name__ == "__main__":
    main()
