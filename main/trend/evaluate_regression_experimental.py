"""Evaluate conditional trend-v3 regression on the 113-row experimental set."""

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
from pymatgen.core import Composition
from sklearn.metrics import accuracy_score, confusion_matrix

from main.paths import DATA_DIR, portable_path
from main.features import normalize_family
from main.trend.features import _formula_descriptor_cache, _pair_numeric_features
from main.trend.predict import _grouped_raw_rows, _parse_conductivity, _parse_formula
from main.trend.predict_regression import DEFAULT_MODEL, _matrix
from main.trend.regression_v3 import (
    BASELINE_LOG10_COLUMN,
    DEFAULT_RUN_DIR,
    LOG_RATIO_COLUMN,
    MODEL_FEATURE_COLUMNS_V3,
    SPLIT_GROUP_COLUMN,
    TARGET_COLUMN,
    WEIGHT_COLUMN,
    write_json,
)
from main.trend.train_pair_regression_v3 import _metrics


DEFAULT_RAW = DATA_DIR / "experimental" / "raw" / "experimental-data.csv"
DEFAULT_ANNOTATIONS = (
    DATA_DIR / "experimental" / "annotations" / "experimental-data-labeled.csv"
)
DEFAULT_OUTPUT_DIR = DEFAULT_RUN_DIR / "external_evaluation" / "experimental_113"
ABSOLUTE_DIRECTION_THRESHOLD_S_CM = 1e-4
DIRECTION_LABELS = ("decrease", "unchanged", "increase")


def _direction(values: np.ndarray, threshold: float) -> np.ndarray:
    return np.where(
        values > threshold,
        "increase",
        np.where(values < -threshold, "decrease", "unchanged"),
    )


def _reduced_formula(formula: str) -> str:
    return Composition(_parse_formula(formula)).reduced_formula


def build_experimental_pairs(
    raw_path: Path = DEFAULT_RAW,
    annotation_path: Path = DEFAULT_ANNOTATIONS,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build adjacent pairs without crossing manually annotated block boundaries."""

    annotations = pd.read_csv(annotation_path, dtype=str, keep_default_na=False)
    required = {"ID", "True Composition", "Family"}
    missing = sorted(required - set(annotations.columns))
    if missing:
        raise ValueError(f"Annotations are missing columns: {missing}")
    if annotations["ID"].duplicated().any():
        raise ValueError("Annotation IDs must be unique.")
    annotation_by_id = annotations.set_index("ID")

    raw_rows = _grouped_raw_rows(raw_path)
    if len(raw_rows) != len(annotations):
        raise ValueError(
            "Raw/annotation row count mismatch: "
            f"raw={len(raw_rows)}, annotations={len(annotations)}"
        )
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in raw_rows:
        row_id = row["id"]
        if row_id not in annotation_by_id.index:
            raise ValueError(f"Raw ID is absent from annotations: {row_id}")
        annotation = annotation_by_id.loc[row_id]
        if row["formula"] != annotation["True Composition"]:
            raise ValueError(
                f"Formula mismatch for {row_id}: raw={row['formula']!r}, "
                f"annotation={annotation['True Composition']!r}"
            )
        grouped.setdefault(row["group_segment"], []).append(
            {
                **row,
                "family": str(annotation["Family"]),
                "conductivity_S_cm-1": _parse_conductivity(
                    row["conductivity"], 1.0
                ),
            }
        )

    records: list[dict[str, Any]] = []
    for segment, rows in grouped.items():
        unique_group_id = f"{rows[0]['group_id']}:segment_{segment}"
        for position, (left, right) in enumerate(zip(rows, rows[1:]), start=1):
            sigma_a = float(left["conductivity_S_cm-1"])
            sigma_b = float(right["conductivity_S_cm-1"])
            if sigma_a <= 0 or sigma_b <= 0:
                raise ValueError("Experimental conductivities must be positive.")
            records.append(
                {
                    "pair_id": f"exp_pair_{len(records) + 1:04d}",
                    "group_id": unique_group_id,
                    "display_group_id": left["group_id"],
                    "group_segment": segment,
                    "position_in_group": position,
                    "id_a": left["id"],
                    "id_b": right["id"],
                    "formula_a": left["formula"],
                    "formula_b": right["formula"],
                    "model_formula_a": _parse_formula(left["formula"]),
                    "model_formula_b": _parse_formula(right["formula"]),
                    "family_a": left["family"],
                    "family_b": right["family"],
                    "conductivity_a_S_cm-1": sigma_a,
                    "conductivity_b_S_cm-1": sigma_b,
                    BASELINE_LOG10_COLUMN: math.log10(sigma_a),
                    TARGET_COLUMN: math.log10(sigma_b),
                    LOG_RATIO_COLUMN: math.log10(sigma_b) - math.log10(sigma_a),
                    "delta_conductivity_S_cm-1": sigma_b - sigma_a,
                }
            )
    pairs = pd.DataFrame.from_records(records)
    pairs[SPLIT_GROUP_COLUMN] = pairs["group_id"]
    pairs[WEIGHT_COLUMN] = 1.0 / pairs.groupby("group_id")["pair_id"].transform(
        "size"
    )
    audit = {
        "source_rows": int(len(raw_rows)),
        "group_segments": int(len(grouped)),
        "pair_rows": int(len(pairs)),
        "pairing": "adjacent rows within each manually annotated group segment",
        "comparability_assumption": (
            "all rows were produced by one laboratory and synthesis/measurement "
            "conditions are treated as consistent"
        ),
    }
    return pairs, audit


def _training_overlap(pairs: pd.DataFrame, model_path: Path) -> dict[str, Any]:
    data_dir = model_path.parents[1] / "data"
    training_paths = [data_dir / "train.csv", data_dir / "validation.csv"]
    missing = [path for path in training_paths if not path.exists()]
    if missing:
        return {
            "available": False,
            "reason": f"Training tables not found: {[str(path) for path in missing]}",
        }
    training = pd.concat(
        [pd.read_csv(path, dtype=str, keep_default_na=False) for path in training_paths],
        ignore_index=True,
    )
    training_formulas = set(training["reduced_formula_a"]) | set(
        training["reduced_formula_b"]
    )
    training_pairs = {
        tuple(sorted((left, right)))
        for left, right in zip(
            training["reduced_formula_a"], training["reduced_formula_b"]
        )
    }
    reduced_a = pairs["model_formula_a"].map(_reduced_formula)
    reduced_b = pairs["model_formula_b"].map(_reduced_formula)
    pair_overlap = np.array(
        [
            tuple(sorted((left, right))) in training_pairs
            for left, right in zip(reduced_a, reduced_b)
        ]
    )
    formula_overlap = reduced_a.isin(training_formulas) | reduced_b.isin(
        training_formulas
    )
    return {
        "available": True,
        "training_scope": "final model train + validation splits",
        "pairs_with_any_formula_overlap": int(formula_overlap.sum()),
        "exact_unordered_pair_overlap": int(pair_overlap.sum()),
        "unique_experimental_formulas_overlapping": int(
            len((set(reduced_a) | set(reduced_b)) & training_formulas)
        ),
        "doi_overlap": "not auditable because experimental_113 has no DOI field",
    }


def evaluate_experimental_regression(
    model_path: Path = DEFAULT_MODEL,
    raw_path: Path = DEFAULT_RAW,
    annotation_path: Path = DEFAULT_ANNOTATIONS,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Evaluate the trained conditional model and write predictions and metrics."""

    pairs, data_audit = build_experimental_pairs(raw_path, annotation_path)
    formulas = pd.unique(
        pd.concat(
            [pairs["model_formula_a"], pairs["model_formula_b"]],
            ignore_index=True,
        )
    )
    cache = _formula_descriptor_cache(formulas, show_progress=True)
    feature_rows = [
        _pair_numeric_features(cache[left], cache[right], left, right)
        for left, right in zip(pairs["model_formula_a"], pairs["model_formula_b"])
    ]
    features = pd.DataFrame.from_records(feature_rows)
    features[BASELINE_LOG10_COLUMN] = pairs[BASELINE_LOG10_COLUMN].to_numpy(float)
    features["family"] = pairs["family_a"].map(normalize_family).to_numpy()

    bundle = joblib.load(model_path)
    expected_features = list(bundle.get("numeric_features", []))
    if expected_features != MODEL_FEATURE_COLUMNS_V3:
        raise ValueError(
            "Model is not the conditional F43 schema required for this evaluation."
        )
    prediction_log_sigma_b = np.asarray(
        bundle["model"].predict(_matrix(features, bundle)), dtype=float
    )
    feature_min = pd.Series(bundle.get("feature_min", {}), dtype=float).reindex(
        expected_features
    )
    feature_max = pd.Series(bundle.get("feature_max", {}), dtype=float).reindex(
        expected_features
    )
    numeric_features = features[expected_features].apply(pd.to_numeric, errors="coerce")
    outside_matrix = numeric_features.lt(feature_min, axis="columns") | numeric_features.gt(
        feature_max, axis="columns"
    )
    metrics_frame = pairs.copy()
    regression_metrics = _metrics(metrics_frame, prediction_log_sigma_b)
    zero_prediction = pairs[BASELINE_LOG10_COLUMN].to_numpy(float)
    zero_metrics = _metrics(metrics_frame, zero_prediction)

    predicted_sigma_b = np.power(10.0, prediction_log_sigma_b)
    sigma_a = pairs["conductivity_a_S_cm-1"].to_numpy(float)
    predicted_delta = predicted_sigma_b - sigma_a
    predicted_log_ratio = prediction_log_sigma_b - pairs[
        BASELINE_LOG10_COLUMN
    ].to_numpy(float)
    true_direction = _direction(
        pairs["delta_conductivity_S_cm-1"].to_numpy(float),
        ABSOLUTE_DIRECTION_THRESHOLD_S_CM,
    )
    predicted_direction = _direction(
        predicted_delta, ABSOLUTE_DIRECTION_THRESHOLD_S_CM
    )

    output = pairs.copy()
    output["predicted_log10_conductivity_b"] = prediction_log_sigma_b
    output["predicted_conductivity_b_S_cm-1"] = predicted_sigma_b
    output["predicted_delta_conductivity_S_cm-1"] = predicted_delta
    output["predicted_absolute_delta_conductivity_S_cm-1"] = np.abs(
        predicted_delta
    )
    output["predicted_delta_log10_ratio"] = predicted_log_ratio
    output["predicted_ratio_b_over_a"] = np.power(10.0, predicted_log_ratio)
    output["absolute_error_delta_conductivity_S_cm-1"] = np.abs(
        predicted_sigma_b - pairs["conductivity_b_S_cm-1"].to_numpy(float)
    )
    output["absolute_error_delta_log10_ratio"] = np.abs(
        prediction_log_sigma_b - pairs[TARGET_COLUMN].to_numpy(float)
    )
    output["true_direction_at_1e-4_S_cm"] = true_direction
    output["predicted_direction_at_1e-4_S_cm"] = predicted_direction
    output["direction_correct_at_1e-4_S_cm"] = (
        true_direction == predicted_direction
    )
    output["features_outside_training_range_count"] = outside_matrix.sum(axis=1)
    output["features_outside_training_range"] = outside_matrix.apply(
        lambda row: ";".join(row.index[row].tolist()), axis=1
    )

    overlap = _training_overlap(pairs, model_path)
    regression_metrics[
        "zero_change_baseline_weighted_mae_delta_log10_ratio"
    ] = zero_metrics["weighted_mae_delta_log10_ratio"]
    regression_metrics["beats_zero_change_baseline_weighted_mae"] = (
        regression_metrics["weighted_mae_delta_log10_ratio"]
        < zero_metrics["weighted_mae_delta_log10_ratio"]
    )
    direction_metrics = {
        "threshold_S_cm-1": ABSOLUTE_DIRECTION_THRESHOLD_S_CM,
        "accuracy": float(accuracy_score(true_direction, predicted_direction)),
        "majority_class_baseline_accuracy": float(
            max(np.sum(true_direction == label) for label in DIRECTION_LABELS)
            / len(true_direction)
        ),
        "zero_change_baseline_accuracy": float(
            np.mean(true_direction == "unchanged")
        ),
        "labels": {
            label: int(np.sum(true_direction == label)) for label in DIRECTION_LABELS
        },
        "predicted_labels": {
            label: int(np.sum(predicted_direction == label))
            for label in DIRECTION_LABELS
        },
        "confusion_matrix_labels": list(DIRECTION_LABELS),
        "confusion_matrix": confusion_matrix(
            true_direction, predicted_direction, labels=DIRECTION_LABELS
        ).tolist(),
    }
    feature_range_audit = {
        "pairs_outside_any_training_feature_range": int(
            outside_matrix.any(axis=1).sum()
        ),
        "fraction_outside_any_training_feature_range": float(
            outside_matrix.any(axis=1).mean()
        ),
        "most_common_outside_features": {
            str(name): int(count)
            for name, count in outside_matrix.sum(axis=0)
            .sort_values(ascending=False)
            .items()
            if count > 0
        },
    }
    group_rows = []
    for group_id, group in output.groupby("group_id", sort=False):
        group_rows.append(
            {
                "group_id": group_id,
                "pairs": int(len(group)),
                "mae_delta_log10_ratio": float(
                    group["absolute_error_delta_log10_ratio"].mean()
                ),
                "mae_delta_conductivity_S_cm-1": float(
                    group["absolute_error_delta_conductivity_S_cm-1"].mean()
                ),
                "direction_accuracy_at_1e-4_S_cm": float(
                    group["direction_correct_at_1e-4_S_cm"].mean()
                ),
            }
        )
    group_metrics = pd.DataFrame(group_rows)
    metrics = {
        "model": portable_path(model_path),
        "model_status": bundle.get("model_status", "unknown"),
        "data": data_audit,
        "training_overlap_audit": overlap,
        "feature_range_audit": feature_range_audit,
        "regression": regression_metrics,
        "zero_change_baseline": zero_metrics,
        "absolute_direction": direction_metrics,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_dir / "predictions.csv", index=False)
    group_metrics.to_csv(output_dir / "group_metrics.csv", index=False)
    write_json(output_dir / "metrics.json", metrics)
    report = [
        "# Trend-v3 conditional regression on experimental 113",
        "",
        f"- Source rows: {data_audit['source_rows']}",
        f"- Adjacent pairs: {data_audit['pair_rows']}",
        "- Comparability: one laboratory; synthesis and measurement conditions treated as consistent",
        "- DOI independence: not auditable because the external table has no DOI",
        f"- Formula-overlap pairs: {overlap.get('pairs_with_any_formula_overlap', 'unknown')}",
        f"- Exact pair overlap: {overlap.get('exact_unordered_pair_overlap', 'unknown')}",
        f"- Pairs outside feature range: {feature_range_audit['pairs_outside_any_training_feature_range']}",
        "",
        "## Regression metrics",
        "",
        *[f"- `{key}`: {value}" for key, value in regression_metrics.items()],
        "",
        "## Absolute-change direction",
        "",
        *[f"- `{key}`: {value}" for key, value in direction_metrics.items()],
        "",
        "## Group metrics",
        "",
        group_metrics.to_markdown(index=False),
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    result = evaluate_experimental_regression(
        model_path=args.model,
        raw_path=args.raw,
        annotation_path=args.annotations,
        output_dir=args.output_dir,
    )
    print(f"Source rows: {result['data']['source_rows']}")
    print(f"Adjacent pairs: {result['data']['pair_rows']}")
    print(
        "Weighted log-ratio MAE: "
        f"{result['regression']['weighted_mae_delta_log10_ratio']:.4f}"
    )
    print(f"Output: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
