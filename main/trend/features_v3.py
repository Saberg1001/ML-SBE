"""Build the feature set for trend-v3 regression.

The v3 feature set extends the existing classification feature set by one
additional input: ``log10_IC_a``, the log-base-10 conductivity of the baseline
material.  Adding the baseline is physically motivated: the magnitude of an
achievable Δlog10 is bounded by where the material starts, and the model needs
that context to give calibrated predictions.

All composition-difference features come from the existing
``main.trend.features`` module so the descriptor definitions stay in one place.
The only new column is ``log10_IC_a``, appended at the end.

Target: ``delta_log10_IC = log10(IC_B) - log10(IC_A)``
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

if __package__ is None:
    _FILE = Path(__file__).resolve()
    if str(_FILE.parents[2]) not in sys.path:
        sys.path.insert(0, str(_FILE.parents[2]))
    __package__ = f"{_FILE.parents[1].name}.{_FILE.parents[0].name}"

import math

import numpy as np
import pandas as pd

from ..paths import DATA_DIR
from .features import (
    MODEL_FEATURE_COLUMNS,
    OPTIONAL_MODEL_FEATURE_COLUMNS,
    FeatureComputationError,
    _formula_descriptor_cache,
    _pair_numeric_features,
)


PAIRS_INPUT_V3 = DATA_DIR / "trend" / "data-trend-v3-pairs.csv"
DEFAULT_OUTPUT_V3 = DATA_DIR / "trend" / "data-trend-v3-pairs-feature.csv"

# v3 model features = existing composition-difference features + baseline IC.
MODEL_FEATURE_COLUMNS_V3: list[str] = [*MODEL_FEATURE_COLUMNS, "log10_IC_a"]

TARGET_COLUMN = "delta_log10_IC"

# Metadata columns written alongside features for training and audit.
# Note: ``log10_IC_a`` is intentionally excluded here — it is a model feature
# (part of MODEL_FEATURE_COLUMNS_V3) and must be written only once, from the
# feature block.  Including it in both blocks would duplicate the column.
TRACE_COLUMNS_V3 = [
    "group_id",
    "pair_id",
    "pairing_strategy",
    "formula_a",
    "formula_b",
    "conductivity_a_S_cm-1",
    "conductivity_b_S_cm-1",
    "log10_IC_b",
    TARGET_COLUMN,
    "family",
    "doi",
    "id_a",
    "id_b",
    "pair_weight_group_equal",
]

# Split output columns (same columns written to train/validation CSVs).
OUTPUT_COLUMNS_V3 = [
    "group_id",
    "pair_id",
    "formula_a",
    "formula_b",
    "family",
    TARGET_COLUMN,
    "pair_weight_group_equal",
    "doi",
    *MODEL_FEATURE_COLUMNS_V3,
]


def build_v3_feature_table(pairs: pd.DataFrame) -> pd.DataFrame:
    """Compute v3 features from a pair table produced by ``pairing_v3``.

    Required columns: ``group_id``, ``pair_id``, ``formula_a``, ``formula_b``,
    ``log10_IC_a``, ``delta_log10_IC``, ``family``, ``doi``.
    """
    required = {
        "group_id", "pair_id", "formula_a", "formula_b",
        "log10_IC_a", "delta_log10_IC", "family", "doi",
    }
    missing = sorted(required - set(pairs.columns))
    if missing:
        raise ValueError(f"Pair table is missing columns: {missing}")

    frame = pairs.copy().reset_index(drop=True)
    fa = frame["formula_a"].astype(str).str.strip()
    fb = frame["formula_b"].astype(str).str.strip()

    cache = _formula_descriptor_cache(
        pd.concat([fa, fb], ignore_index=True), show_progress=True
    )

    records: list[dict] = []
    for a, b in zip(fa, fb):
        records.append(_pair_numeric_features(cache[a], cache[b], a, b))

    numeric = pd.DataFrame.from_records(records, columns=MODEL_FEATURE_COLUMNS)
    numeric.index = frame.index

    # Append the baseline log10 conductivity — the one new v3 feature.
    log10_ic_a = pd.to_numeric(frame["log10_IC_a"], errors="raise")
    if not np.isfinite(log10_ic_a.to_numpy(dtype=float)).all():
        raise FeatureComputationError("log10_IC_a contains non-finite values.")
    numeric["log10_IC_a"] = log10_ic_a.to_numpy(dtype=float)

    # Pair-equal weight within each group.
    pair_weight = 1.0 / frame.groupby("group_id")["pair_id"].transform("size")

    output_cols = [c for c in TRACE_COLUMNS_V3 if c in frame.columns]
    output = frame[output_cols].copy()
    output["pair_weight_group_equal"] = pair_weight.to_numpy(dtype=float)

    # Drop log10_IC_a from trace (it will come from the feature matrix) if
    # it was written there, then concatenate.
    feature_block = numeric[MODEL_FEATURE_COLUMNS_V3]
    result = pd.concat(
        [output.reset_index(drop=True), feature_block.reset_index(drop=True)],
        axis=1,
    )

    _validate_v3_feature_table(result)
    return result


def _validate_v3_feature_table(frame: pd.DataFrame) -> None:
    missing = sorted(set(MODEL_FEATURE_COLUMNS_V3) - set(frame.columns))
    if missing:
        raise ValueError(f"v3 feature table is missing model columns: {missing}")
    if TARGET_COLUMN not in frame.columns:
        raise ValueError(f"v3 feature table is missing target column '{TARGET_COLUMN}'.")

    numeric = frame[MODEL_FEATURE_COLUMNS_V3].apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=float)
    if np.isinf(values).any():
        bad = numeric.columns[np.isinf(values).any(axis=0)].tolist()
        raise ValueError(f"Infinite values in v3 model features: {bad}")

    optional = OPTIONAL_MODEL_FEATURE_COLUMNS
    forbidden_nan = [
        col for col in numeric.columns
        if col not in optional and numeric[col].isna().any()
    ]
    if forbidden_nan:
        raise ValueError(f"Unexpected NaN in v3 model features: {forbidden_nan}")

    delta = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce")
    if not np.isfinite(delta.to_numpy(dtype=float)).all():
        raise ValueError(f"Target column '{TARGET_COLUMN}' contains non-finite values.")

    if "pair_id" in frame.columns and frame["pair_id"].duplicated().any():
        raise ValueError("pair_id must be unique in the v3 feature table.")


def main() -> None:
    """Build v3 pair features from the v3 pairs table (point-run entry).

    Run directly via ``python main/trend/features_v3.py`` (the "Run" button).
    Input  : data/trend/data-trend-v3-pairs.csv
    Output : data/trend/data-trend-v3-pairs-feature.csv
    """
    pairs = pd.read_csv(PAIRS_INPUT_V3, dtype=str, keep_default_na=False)
    output = build_v3_feature_table(pairs)
    DEFAULT_OUTPUT_V3.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(DEFAULT_OUTPUT_V3, index=False)
    print(
        f"pair_rows={len(output)}  "
        f"feature_columns={len(MODEL_FEATURE_COLUMNS_V3)}  "
        f"(composition_diff={len(MODEL_FEATURE_COLUMNS)}, log10_IC_a=1)"
    )
    print(f"Output CSV : {DEFAULT_OUTPUT_V3.resolve()}")


if __name__ == "__main__":
    main()
