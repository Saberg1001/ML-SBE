"""Predict conditional conductivity and derived changes for one material pair."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
from typing import Any

if __package__ is None:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

import joblib
import numpy as np
import pandas as pd

from main.features import normalize_family
from main.trend.features import _formula_descriptor_cache, _pair_numeric_features
from main.trend.regression_v3 import (
    BASELINE_LOG10_COLUMN,
    DEFAULT_RUN_DIR,
    MODEL_FEATURE_COLUMNS_V3,
)


DEFAULT_MODEL = DEFAULT_RUN_DIR / "best_model" / "model.joblib"


def _feature_row(
    formula_a: str,
    formula_b: str,
    conductivity_a_s_cm: float,
    family: str = "unknown",
) -> pd.DataFrame:
    conductivity_a_s_cm = float(conductivity_a_s_cm)
    if not math.isfinite(conductivity_a_s_cm) or conductivity_a_s_cm <= 0:
        raise ValueError("conductivity_a_s_cm must be finite and greater than zero.")
    cache = _formula_descriptor_cache([formula_a, formula_b])
    values = _pair_numeric_features(
        cache[formula_a], cache[formula_b], formula_a, formula_b
    )
    values[BASELINE_LOG10_COLUMN] = math.log10(conductivity_a_s_cm)
    values["family"] = normalize_family(family)
    return pd.DataFrame([values], columns=[*MODEL_FEATURE_COLUMNS_V3, "family"])


def _matrix(frame: pd.DataFrame, bundle: dict[str, Any]) -> np.ndarray:
    numeric_features = list(bundle.get("numeric_features", MODEL_FEATURE_COLUMNS_V3))
    medians = (
        pd.Series(bundle["numeric_medians"], dtype=float)
        .reindex(numeric_features)
        .fillna(0.0)
    )
    numeric = (
        frame[numeric_features]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(medians)
        .to_numpy(np.float32)
    )
    categories = [str(value) for value in bundle.get("family_categories", [])]
    if not categories:
        return numeric
    family = frame.get("family", pd.Series("unknown", index=frame.index))
    family = family.map(normalize_family).astype(str)
    one_hot = pd.get_dummies(family).reindex(columns=categories, fill_value=0)
    return np.concatenate([numeric, one_hot.to_numpy(np.float32)], axis=1)


def predict_conductivity_delta(
    formula_a: str,
    formula_b: str,
    conductivity_a_s_cm: float,
    model_path: str | Path = DEFAULT_MODEL,
    family: str = "unknown",
) -> dict[str, Any]:
    """Return sigma_B, absolute change, log-ratio change, and intervals."""

    formula_a = str(formula_a).strip()
    formula_b = str(formula_b).strip()
    if not formula_a or not formula_b:
        raise ValueError("Both formulas are required.")
    conductivity_a_s_cm = float(conductivity_a_s_cm)
    if not math.isfinite(conductivity_a_s_cm) or conductivity_a_s_cm <= 0:
        raise ValueError("conductivity_a_s_cm must be finite and greater than zero.")
    bundle = joblib.load(Path(model_path))
    feature_frame = _feature_row(
        formula_a, formula_b, conductivity_a_s_cm, family=family
    )
    predicted_log_sigma_b = float(
        bundle["model"].predict(_matrix(feature_frame, bundle))[0]
    )
    predicted_sigma_b = float(10.0**predicted_log_sigma_b)
    baseline_log_sigma_a = math.log10(conductivity_a_s_cm)
    delta_log_ratio = predicted_log_sigma_b - baseline_log_sigma_a
    ratio = float(10.0**delta_log_ratio)
    delta_conductivity = predicted_sigma_b - conductivity_a_s_cm
    interval = float(
        bundle.get(
            "prediction_interval_absolute_error_delta_log10_ratio_90",
            math.nan,
        )
    )
    lower_log_sigma_b = predicted_log_sigma_b - interval
    upper_log_sigma_b = predicted_log_sigma_b + interval
    lower_sigma_b = float(10.0**lower_log_sigma_b)
    upper_sigma_b = float(10.0**upper_log_sigma_b)
    lower_delta_log_ratio = delta_log_ratio - interval
    upper_delta_log_ratio = delta_log_ratio + interval
    feature_min = pd.Series(bundle.get("feature_min", {}), dtype=float)
    feature_max = pd.Series(bundle.get("feature_max", {}), dtype=float)
    numeric = feature_frame.loc[0, MODEL_FEATURE_COLUMNS_V3].astype(float)
    common = numeric.index.intersection(feature_min.index).intersection(feature_max.index)
    outside = [
        name for name in common
        if numeric[name] < feature_min[name] or numeric[name] > feature_max[name]
    ]
    if delta_log_ratio > 0.1:
        direction = "increase"
    elif delta_log_ratio < -0.1:
        direction = "decrease"
    else:
        direction = "approximately_unchanged"
    return {
        "formula_a": formula_a,
        "formula_b": formula_b,
        "conductivity_a_S_cm-1": conductivity_a_s_cm,
        "family": normalize_family(family),
        "predicted_log10_conductivity_b": predicted_log_sigma_b,
        "predicted_conductivity_b_S_cm-1": predicted_sigma_b,
        "predicted_delta_conductivity_S_cm-1": delta_conductivity,
        "predicted_absolute_delta_conductivity_S_cm-1": abs(delta_conductivity),
        "predicted_delta_log10_ratio": delta_log_ratio,
        "predicted_ratio_b_over_a": ratio,
        "predicted_direction_at_0.1_log10": direction,
        "prediction_interval_log10_conductivity_b_90": [
            lower_log_sigma_b,
            upper_log_sigma_b,
        ],
        "prediction_interval_conductivity_b_S_cm-1_90": [
            lower_sigma_b,
            upper_sigma_b,
        ],
        "prediction_interval_delta_conductivity_S_cm-1_90": [
            lower_sigma_b - conductivity_a_s_cm,
            upper_sigma_b - conductivity_a_s_cm,
        ],
        "prediction_interval_delta_log10_ratio_90": [
            lower_delta_log_ratio,
            upper_delta_log_ratio,
        ],
        "prediction_interval_ratio_90": [
            float(10.0**lower_delta_log_ratio),
            float(10.0**upper_delta_log_ratio),
        ],
        "features_outside_training_range": outside,
        "extrapolation_warning": bool(outside),
        "model_status": bundle.get("model_status", "unknown"),
        "deployment_warning": (
            "Current artifact is experimental and did not pass the zero-baseline gate."
            if bundle.get("model_status") == "experimental_not_for_deployment"
            else ""
        ),
        "interpretation": (
            "Expected matched-condition sigma_B given measured sigma_A; "
            "not a direct cross-paper measurement difference."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("formula_a")
    parser.add_argument("formula_b")
    parser.add_argument("conductivity_a_s_cm", type=float)
    parser.add_argument("--family", default="unknown")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()
    result = predict_conductivity_delta(
        args.formula_a,
        args.formula_b,
        args.conductivity_a_s_cm,
        args.model,
        args.family,
    )
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
