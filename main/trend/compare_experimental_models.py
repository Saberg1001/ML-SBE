"""Compare F27, F42, and F54 trend models on grouped experimental pairs.

The summary table is flat, so group boundaries are restored from the two raw
experimental sources before adjacent pairs are built. Repeated group labels are
treated as separate blocks according to their position in the raw files.
"""

from __future__ import annotations

import json
import os
import sys

# Allow direct execution from the project root or an editor run button.
if __package__ is None:
    _PROJECT_ROOT = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from main.trend.features import (
    ALL_COMPUTED_FEATURE_COLUMNS,
    _formula_descriptor_cache,
    _pair_numeric_features,
    classify_trend_delta,
)
from main.trend.predict import _parse_conductivity, _parse_formula, _prediction_matrix


ROOT = Path(__file__).resolve().parents[2]
SUMMARY_PATH = ROOT / "data/experimental/experimental-halide-summary.csv"
GENERAL_RAW_PATH = ROOT / "data/experimental/raw/experimental-data.csv"
HALIDE_RAW_PATH = ROOT / "data/experimental/raw/halide.csv"
OUTPUT_DIR = ROOT / "tmp/exp"
TRAIN_TABLES = (
    ROOT / "data/trend/data-trend-v1-pairs-feature-train.csv",
    ROOT / "data/trend/data-trend-v2-pairs-feature-train.csv",
)
LABELS = ("decrease", "unchanged", "increase")
HIGH_CONDUCTIVITY_S_CM = 1e-3


@dataclass(frozen=True)
class ModelSpec:
    """One immutable model artifact used in the comparison."""

    model_id: str
    feature_count: int
    algorithm: str
    run_name: str
    model_path: Path
    role: str
    selection_basis: str


def _run_model_path(run_name: str, algorithm: str) -> Path:
    return ROOT / "runs/trend" / run_name / algorithm / "model.joblib"


MODEL_SPECS = (
    ModelSpec(
        model_id="f27_groupcv_reference",
        feature_count=27,
        algorithm="catboost",
        run_name="trend_cls_v2_f27_family_abs01_optuna50_seed42",
        model_path=_run_model_path(
            "trend_cls_v2_f27_family_abs01_optuna50_seed42", "catboost"
        ),
        role="grouped_cv_reference",
        selection_basis="newer grouped-CV and swap F27 reference",
    ),
    ModelSpec(
        model_id="f42",
        feature_count=42,
        algorithm="lightgbm",
        run_name="trend_cls_f42_family_dsigma1e-4_swap_groupcv5_optuna50_seed42",
        model_path=_run_model_path(
            "trend_cls_f42_family_dsigma1e-4_swap_groupcv5_optuna50_seed42",
            "lightgbm",
        ),
        role="primary",
        selection_basis="best grouped-CV weighted macro-F1 in the F42 run",
    ),
    ModelSpec(
        model_id="f54",
        feature_count=54,
        algorithm="lightgbm",
        run_name="trend_cls_v2_f54_family_dsigma1e-4_swap_groupcv5_optuna50_seed42",
        model_path=_run_model_path(
            "trend_cls_v2_f54_family_dsigma1e-4_swap_groupcv5_optuna50_seed42",
            "lightgbm",
        ),
        role="primary",
        selection_basis="best grouped-CV weighted macro-F1 in the F54 run",
    ),
    ModelSpec(
        model_id="f42_catboost_control",
        feature_count=42,
        algorithm="catboost",
        run_name="trend_cls_f42_family_dsigma1e-4_swap_groupcv5_optuna50_seed42",
        model_path=_run_model_path(
            "trend_cls_f42_family_dsigma1e-4_swap_groupcv5_optuna50_seed42",
            "catboost",
        ),
        role="same_algorithm_control",
        selection_basis="CatBoost control for feature-set comparison",
    ),
    ModelSpec(
        model_id="f54_catboost_control",
        feature_count=54,
        algorithm="catboost",
        run_name="trend_cls_v2_f54_family_dsigma1e-4_swap_groupcv5_optuna50_seed42",
        model_path=_run_model_path(
            "trend_cls_v2_f54_family_dsigma1e-4_swap_groupcv5_optuna50_seed42",
            "catboost",
        ),
        role="same_algorithm_control",
        selection_basis="CatBoost control for feature-set comparison",
    ),
    ModelSpec(
        model_id="f27",
        feature_count=27,
        algorithm="catboost",
        run_name="trend_cls_v3_f27_family_abs01_fixedval_optuna10_seed42",
        model_path=_run_model_path(
            "trend_cls_v3_f27_family_abs01_fixedval_optuna10_seed42", "catboost"
        ),
        role="primary",
        selection_basis="user-referenced fixed-validation model with 59.7% validation accuracy",
    ),
)

PRIMARY_MODEL_IDS = ("f27", "f42", "f54")
CATBOOST_CONTROL_IDS = (
    "f27_groupcv_reference",
    "f42_catboost_control",
    "f54_catboost_control",
)
F27_PROTOCOL_IDS = ("f27", "f27_groupcv_reference")


def _read_raw_segments(raw_path: Path, dataset: str) -> list[dict]:
    """Read ordered raw blocks while preserving repeated group labels."""
    segments: list[dict] = []
    current_segment: dict | None = None
    group_occurrences: Counter[str] = Counter()
    halide_index = 0

    for line_number, line in enumerate(
        raw_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        parts = line.split("\t")
        first = parts[0].strip() if parts else ""
        if not first:
            continue

        is_group_header = first.lower().startswith("group") and all(
            not value.strip() for value in parts[1:]
        )
        if is_group_header:
            group_occurrences[first] += 1
            current_segment = {
                "dataset": dataset,
                "group_id": first,
                "group_occurrence": group_occurrences[first],
                "group_segment_id": (
                    f"{dataset}:{first}:{group_occurrences[first]}"
                ),
                "rows": [],
            }
            segments.append(current_segment)
            continue

        if dataset == "general_experimental":
            if not first.startswith("exp_") or len(parts) < 3:
                continue
            row_id = first
            formula = parts[1].strip()
            conductivity_text = parts[2].strip()
        elif dataset == "halide":
            if current_segment is None or len(parts) < 2:
                continue
            halide_index += 1
            row_id = f"hal_{halide_index:03d}"
            formula = first
            conductivity_text = parts[1].strip()
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")

        if current_segment is None:
            raise ValueError(
                f"Data row before a group header in {raw_path}:{line_number}"
            )
        conductivity = _parse_conductivity(conductivity_text, 1.0)
        current_segment["rows"].append(
            {
                "id": row_id,
                "formula": formula,
                "conductivity_raw_S_cm-1": conductivity,
                "raw_line": line_number,
            }
        )

    empty = [segment["group_segment_id"] for segment in segments if not segment["rows"]]
    if empty:
        raise ValueError(f"Empty raw group segments in {raw_path}: {empty}")
    return segments


def _load_summary() -> pd.DataFrame:
    """Load and validate the flat experimental summary."""
    summary = pd.read_csv(SUMMARY_PATH, dtype=str, keep_default_na=False)
    required = {"ID", "化学式", "电导率（S/cm）", "制备方式", "family"}
    missing = sorted(required - set(summary.columns))
    if missing:
        raise ValueError(f"Summary is missing columns: {missing}")
    if summary["ID"].duplicated().any():
        duplicates = summary.loc[summary["ID"].duplicated(), "ID"].tolist()
        raise ValueError(f"Duplicate summary IDs: {duplicates}")
    summary["conductivity_S_cm-1"] = pd.to_numeric(
        summary["电导率（S/cm）"], errors="raise"
    )
    if not np.isfinite(summary["conductivity_S_cm-1"]).all():
        raise ValueError("Summary contains non-finite conductivity values")
    if (summary["conductivity_S_cm-1"] < 0).any():
        raise ValueError("Summary contains negative conductivity values")
    return summary


def _build_pairs(summary: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Restore group blocks and build only within-block adjacent pairs."""
    segments = [
        *_read_raw_segments(GENERAL_RAW_PATH, "general_experimental"),
        *_read_raw_segments(HALIDE_RAW_PATH, "halide"),
    ]
    summary_by_id = summary.set_index("ID", drop=False)
    raw_ids: list[str] = []
    source_counts: Counter[str] = Counter()

    for segment in segments:
        source_counts[segment["dataset"]] += len(segment["rows"])
        for row in segment["rows"]:
            row_id = row["id"]
            raw_ids.append(row_id)
            if row_id not in summary_by_id.index:
                raise ValueError(f"Raw ID is absent from summary: {row_id}")
            summary_row = summary_by_id.loc[row_id]
            if row["formula"] != summary_row["化学式"]:
                raise ValueError(
                    f"Formula mismatch for {row_id}: raw={row['formula']!r}, "
                    f"summary={summary_row['化学式']!r}"
                )
            if not np.isclose(
                row["conductivity_raw_S_cm-1"],
                float(summary_row["conductivity_S_cm-1"]),
                rtol=1e-10,
                atol=1e-12,
            ):
                raise ValueError(
                    f"Conductivity mismatch for {row_id}: "
                    f"raw={row['conductivity_raw_S_cm-1']}, "
                    f"summary={summary_row['conductivity_S_cm-1']}"
                )

    missing_ids = sorted(set(summary["ID"]) - set(raw_ids))
    extra_ids = sorted(set(raw_ids) - set(summary["ID"]))
    if missing_ids or extra_ids or len(raw_ids) != len(summary):
        raise ValueError(
            "Raw-to-summary coverage mismatch: "
            f"missing={missing_ids}, extra={extra_ids}, "
            f"raw_rows={len(raw_ids)}, summary_rows={len(summary)}"
        )

    records: list[dict] = []
    pair_counts: Counter[str] = Counter()
    for segment in segments:
        dataset = segment["dataset"]
        for position, (left, right) in enumerate(
            zip(segment["rows"], segment["rows"][1:]), start=1
        ):
            pair_counts[dataset] += 1
            left_summary = summary_by_id.loc[left["id"]]
            right_summary = summary_by_id.loc[right["id"]]
            family_a = str(left_summary["family"])
            family_b = str(right_summary["family"])
            method_a = str(left_summary["制备方式"])
            method_b = str(right_summary["制备方式"])
            formula_a = str(left_summary["化学式"])
            formula_b = str(right_summary["化学式"])
            sigma_a = float(left_summary["conductivity_S_cm-1"])
            sigma_b = float(right_summary["conductivity_S_cm-1"])
            records.append(
                {
                    "dataset": dataset,
                    "pair_id": (
                        f"exp_pair_{pair_counts[dataset]:04d}"
                        if dataset == "general_experimental"
                        else f"hal_pair_{pair_counts[dataset]:04d}"
                    ),
                    "group_id": segment["group_id"],
                    "group_occurrence": segment["group_occurrence"],
                    "group_segment_id": segment["group_segment_id"],
                    "position_in_group": position,
                    "id_a": left["id"],
                    "id_b": right["id"],
                    "formula_a": formula_a,
                    "formula_b": formula_b,
                    "model_formula_a": _parse_formula(formula_a),
                    "model_formula_b": _parse_formula(formula_b),
                    "family_a": family_a,
                    "family_b": family_b,
                    "family": family_a if family_a == family_b else "mixed",
                    "preparation_method_a": method_a,
                    "preparation_method_b": method_b,
                    "conductivity_a_S_cm-1": sigma_a,
                    "conductivity_b_S_cm-1": sigma_b,
                }
            )

    pairs = pd.DataFrame(records)
    pairs["delta_conductivity_S_cm-1"] = (
        pairs["conductivity_b_S_cm-1"] - pairs["conductivity_a_S_cm-1"]
    )
    pairs["absolute_delta_conductivity_S_cm-1"] = pairs[
        "delta_conductivity_S_cm-1"
    ].abs()
    pairs["max_conductivity_S_cm-1"] = pairs[
        ["conductivity_a_S_cm-1", "conductivity_b_S_cm-1"]
    ].max(axis=1)
    pairs["min_conductivity_S_cm-1"] = pairs[
        ["conductivity_a_S_cm-1", "conductivity_b_S_cm-1"]
    ].min(axis=1)
    pairs["true_label"] = classify_trend_delta(
        pairs["delta_conductivity_S_cm-1"].to_numpy(float)
    )
    pairs["additive_omitted_from_features"] = (
        (pairs["formula_a"] != pairs["model_formula_a"])
        | (pairs["formula_b"] != pairs["model_formula_b"])
    )

    audit = {
        "summary_rows": int(len(summary)),
        "raw_rows": {key: int(value) for key, value in source_counts.items()},
        "group_segments": int(len(segments)),
        "group_segments_by_dataset": {
            dataset: int(sum(segment["dataset"] == dataset for segment in segments))
            for dataset in source_counts
        },
        "pairs": int(len(pairs)),
        "pairs_by_dataset": {
            key: int(value) for key, value in pair_counts.items()
        },
        "true_label_counts": {
            label: int((pairs["true_label"] == label).sum()) for label in LABELS
        },
        "pairs_with_omitted_additive": int(
            pairs["additive_omitted_from_features"].sum()
        ),
    }
    return pairs, audit


def _add_training_overlap_audit(pairs: pd.DataFrame) -> dict:
    """Annotate exact formula and unordered-pair overlap with training rows."""
    training_formulas: set[str] = set()
    training_pairs: set[tuple[str, str]] = set()
    for path in TRAIN_TABLES:
        frame = pd.read_csv(path, usecols=["化学式_a", "化学式_b"], dtype=str)
        formula_a = frame["化学式_a"].map(_parse_formula)
        formula_b = frame["化学式_b"].map(_parse_formula)
        training_formulas.update(formula_a)
        training_formulas.update(formula_b)
        training_pairs.update(
            tuple(sorted((left, right)))
            for left, right in zip(formula_a, formula_b)
        )

    pairs["formula_a_seen_in_training"] = pairs["model_formula_a"].isin(
        training_formulas
    )
    pairs["formula_b_seen_in_training"] = pairs["model_formula_b"].isin(
        training_formulas
    )
    pairs["either_formula_seen_in_training"] = (
        pairs["formula_a_seen_in_training"]
        | pairs["formula_b_seen_in_training"]
    )
    pairs["unordered_pair_seen_in_training"] = [
        tuple(sorted((left, right))) in training_pairs
        for left, right in zip(pairs["model_formula_a"], pairs["model_formula_b"])
    ]
    return {
        "unique_training_formulas": int(len(training_formulas)),
        "unique_training_pairs": int(len(training_pairs)),
        "pairs_with_seen_endpoint": int(
            pairs["either_formula_seen_in_training"].sum()
        ),
        "pairs_with_exact_unordered_pair_overlap": int(
            pairs["unordered_pair_seen_in_training"].sum()
        ),
        "pairs_with_both_endpoints_unseen": int(
            (~pairs["either_formula_seen_in_training"]).sum()
        ),
    }


def _build_feature_frames(
    pairs: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute forward and reversed numeric features once for every model."""
    formula_a = pairs["model_formula_a"]
    formula_b = pairs["model_formula_b"]
    formulas = pd.unique(pd.concat([formula_a, formula_b], ignore_index=True))
    cache = _formula_descriptor_cache(formulas, show_progress=True)
    forward = pd.DataFrame(
        [
            _pair_numeric_features(cache[left], cache[right], left, right)
            for left, right in zip(formula_a, formula_b)
        ]
    )
    reverse = pd.DataFrame(
        [
            _pair_numeric_features(cache[right], cache[left], right, left)
            for left, right in zip(formula_a, formula_b)
        ]
    )
    return forward, reverse


def _load_preprocessing(model_path: Path, bundle: dict) -> dict:
    preprocessing = bundle.get("preprocessing", {})
    preprocessing_path = model_path.with_name("preprocessing.json")
    if not preprocessing and preprocessing_path.exists():
        preprocessing = json.loads(preprocessing_path.read_text(encoding="utf-8"))
    return preprocessing


def _predict_model(
    spec: ModelSpec,
    pairs: pd.DataFrame,
    forward: pd.DataFrame,
    reverse: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """Predict one artifact with its stored feature schema and swap policy."""
    if not spec.model_path.exists():
        raise FileNotFoundError(spec.model_path)
    bundle = joblib.load(spec.model_path)
    model = bundle["model"]
    model_name = str(bundle.get("model_name", "")).lower()
    numeric_features = list(bundle["numeric_features"])
    class_labels = tuple(bundle.get("class_labels", LABELS))
    if model_name != spec.algorithm:
        raise ValueError(
            f"Algorithm mismatch for {spec.model_id}: "
            f"expected={spec.algorithm}, bundle={model_name}"
        )
    if len(numeric_features) != spec.feature_count:
        raise ValueError(
            f"Feature-count mismatch for {spec.model_id}: "
            f"expected={spec.feature_count}, bundle={len(numeric_features)}"
        )
    if class_labels != LABELS:
        raise ValueError(
            f"Class order mismatch for {spec.model_id}: {class_labels}"
        )

    preprocessing = _load_preprocessing(spec.model_path, bundle)
    medians = pd.Series(
        bundle.get("numeric_medians", preprocessing.get("numeric_medians", {})),
        dtype=float,
    )
    family = pairs["family"].astype(str)
    forward_input = forward.copy()
    reverse_input = reverse.copy()
    if "conductivity_a_S_cm-1" in numeric_features:
        forward_input["conductivity_a_S_cm-1"] = pairs[
            "conductivity_a_S_cm-1"
        ].to_numpy(float)
        reverse_input["conductivity_a_S_cm-1"] = pairs[
            "conductivity_b_S_cm-1"
        ].to_numpy(float)

    forward_matrix = _prediction_matrix(
        forward_input,
        family,
        numeric_features,
        model_name,
        preprocessing,
        medians,
    )
    forward_probabilities = np.asarray(model.predict_proba(forward_matrix), dtype=float)
    swap_symmetric = bool(bundle.get("swap_symmetric_prediction", False))
    if swap_symmetric:
        reverse_matrix = _prediction_matrix(
            reverse_input,
            family,
            numeric_features,
            model_name,
            preprocessing,
            medians,
        )
        reverse_probabilities = np.asarray(
            model.predict_proba(reverse_matrix), dtype=float
        )[:, [2, 1, 0]]
        probabilities = (forward_probabilities + reverse_probabilities) / 2.0
    else:
        probabilities = forward_probabilities

    if probabilities.shape != (len(pairs), len(LABELS)):
        raise ValueError(
            f"Unexpected probability shape for {spec.model_id}: "
            f"{probabilities.shape}"
        )
    predicted = np.asarray(LABELS, dtype=object)[probabilities.argmax(axis=1)]
    result = pairs.copy()
    result["predicted_label"] = predicted
    result["correct"] = result["true_label"] == result["predicted_label"]
    for index, label in enumerate(LABELS):
        result[f"probability_{label}"] = probabilities[:, index]

    categories = list(preprocessing.get("family_categories", []))
    unknown_family_mask = ~family.isin(categories) if categories else pd.Series(False, index=family.index)
    metadata = {
        **asdict(spec),
        "model_path": str(spec.model_path.relative_to(ROOT)),
        "numeric_features": numeric_features,
        "swap_symmetric_prediction": swap_symmetric,
        "family_categories": categories,
        "unknown_family_values": sorted(family[unknown_family_mask].unique()),
        "pairs_with_unknown_family": int(unknown_family_mask.sum()),
    }
    return result, metadata


def _scope_masks(pairs: pd.DataFrame) -> dict[str, pd.Series]:
    """Return consistent evaluation subsets for every model."""
    masks: dict[str, pd.Series] = {
        "overall": pd.Series(True, index=pairs.index),
        "dataset_general_experimental": pairs["dataset"].eq(
            "general_experimental"
        ),
        "dataset_halide": pairs["dataset"].eq("halide"),
        "max_sigma_ge_1e-3": pairs["max_conductivity_S_cm-1"].ge(
            HIGH_CONDUCTIVITY_S_CM
        ),
        "both_sigma_ge_1e-3": pairs["min_conductivity_S_cm-1"].ge(
            HIGH_CONDUCTIVITY_S_CM
        ),
        "both_endpoints_unseen_in_training": ~pairs[
            "either_formula_seen_in_training"
        ],
    }
    for family in sorted(pairs["family"].unique()):
        masks[f"family_{family}"] = pairs["family"].eq(family)
    return masks


def _metric_block(frame: pd.DataFrame) -> tuple[dict, list[dict], list[dict]]:
    """Compute aggregate, per-class, and confusion-matrix metrics."""
    true = frame["true_label"].to_numpy(str)
    predicted = frame["predicted_label"].to_numpy(str)
    precision, recall, class_f1, support = precision_recall_fscore_support(
        true,
        predicted,
        labels=LABELS,
        zero_division=0,
    )
    support = support.astype(int)
    observed = support > 0
    severe = (
        ((true == "decrease") & (predicted == "increase"))
        | ((true == "increase") & (predicted == "decrease"))
    )
    changed = true != "unchanged"
    aggregate = {
        "pairs": int(len(frame)),
        "accuracy": float(accuracy_score(true, predicted)),
        "macro_f1": float(
            f1_score(true, predicted, labels=LABELS, average="macro", zero_division=0)
        ),
        "balanced_accuracy": float(recall[observed].mean()) if observed.any() else None,
        "changed_pair_accuracy": (
            float((true[changed] == predicted[changed]).mean()) if changed.any() else None
        ),
        "severe_reversal_count": int(severe.sum()),
        "severe_reversal_rate": float(severe.mean()),
        "severe_reversal_rate_among_changed": (
            float(severe[changed].mean()) if changed.any() else None
        ),
        "true_label_counts": {
            label: int((true == label).sum()) for label in LABELS
        },
        "predicted_label_counts": {
            label: int((predicted == label).sum()) for label in LABELS
        },
        "confusion_matrix_labels": list(LABELS),
        "confusion_matrix": confusion_matrix(
            true, predicted, labels=LABELS
        ).tolist(),
    }
    per_class = [
        {
            "label": label,
            "support": int(support[index]),
            "predicted_count": int((predicted == label).sum()),
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(class_f1[index]),
        }
        for index, label in enumerate(LABELS)
    ]
    matrix = confusion_matrix(true, predicted, labels=LABELS)
    confusion_rows = [
        {
            "true_label": true_label,
            "predicted_label": predicted_label,
            "count": int(matrix[true_index, predicted_index]),
        }
        for true_index, true_label in enumerate(LABELS)
        for predicted_index, predicted_label in enumerate(LABELS)
    ]
    return aggregate, per_class, confusion_rows


def _evaluate_predictions(
    spec: ModelSpec,
    predictions: pd.DataFrame,
    masks: dict[str, pd.Series],
) -> tuple[dict, list[dict], list[dict], list[dict]]:
    """Evaluate one model over all shared scopes."""
    nested: dict[str, dict] = {}
    comparison_rows: list[dict] = []
    per_class_rows: list[dict] = []
    confusion_rows: list[dict] = []
    model_fields = {
        "model_id": spec.model_id,
        "feature_count": spec.feature_count,
        "algorithm": spec.algorithm,
        "role": spec.role,
        "run_name": spec.run_name,
    }
    for scope, mask in masks.items():
        subset = predictions.loc[mask]
        if subset.empty:
            continue
        aggregate, per_class, confusion = _metric_block(subset)
        nested[scope] = aggregate
        comparison_rows.append({**model_fields, "scope": scope, **aggregate})
        per_class_rows.extend(
            {**model_fields, "scope": scope, **row} for row in per_class
        )
        confusion_rows.extend(
            {**model_fields, "scope": scope, **row} for row in confusion
        )
    return nested, comparison_rows, per_class_rows, confusion_rows


def _wide_predictions(
    pairs: pd.DataFrame,
    predictions: dict[str, pd.DataFrame],
    model_ids: Iterable[str],
) -> pd.DataFrame:
    """Place several model outputs side by side on identical pair rows."""
    base_columns = [
        "dataset",
        "pair_id",
        "group_id",
        "group_occurrence",
        "group_segment_id",
        "id_a",
        "id_b",
        "formula_a",
        "formula_b",
        "family",
        "preparation_method_a",
        "preparation_method_b",
        "conductivity_a_S_cm-1",
        "conductivity_b_S_cm-1",
        "delta_conductivity_S_cm-1",
        "true_label",
        "either_formula_seen_in_training",
        "unordered_pair_seen_in_training",
        "additive_omitted_from_features",
    ]
    wide = pairs[base_columns].copy()
    for model_id in model_ids:
        frame = predictions[model_id]
        if not frame["pair_id"].equals(pairs["pair_id"]):
            raise ValueError(f"Pair order differs for model {model_id}")
        wide[f"{model_id}_prediction"] = frame["predicted_label"]
        wide[f"{model_id}_correct"] = frame["correct"]
        for label in LABELS:
            wide[f"{model_id}_probability_{label}"] = frame[
                f"probability_{label}"
            ]
    return wide


def _comparison_subset(
    comparison: pd.DataFrame,
    model_ids: Iterable[str],
    comparison_set: str,
) -> pd.DataFrame:
    order = {model_id: index for index, model_id in enumerate(model_ids)}
    selected = comparison[comparison["model_id"].isin(order)].copy()
    selected.insert(0, "comparison_set", comparison_set)
    selected["model_order"] = selected["model_id"].map(order)
    selected = selected.sort_values(["scope", "model_order"]).drop(
        columns="model_order"
    )
    return selected


def compare_experimental_models() -> dict:
    """Run the full comparison and write all outputs under ``tmp/exp``."""
    print(f"Loading summary: {SUMMARY_PATH.relative_to(ROOT)}", flush=True)
    summary = _load_summary()
    pairs, data_audit = _build_pairs(summary)
    data_audit["training_overlap"] = _add_training_overlap_audit(pairs)
    print(
        "Built grouped pairs: "
        f"{len(pairs)} total "
        f"({data_audit['pairs_by_dataset']})",
        flush=True,
    )

    print("Computing composition features...", flush=True)
    forward, reverse = _build_feature_frames(pairs)
    masks = _scope_masks(pairs)

    predictions: dict[str, pd.DataFrame] = {}
    metadata: dict[str, dict] = {}
    metrics_by_model: dict[str, dict] = {}
    comparison_rows: list[dict] = []
    per_class_rows: list[dict] = []
    confusion_rows: list[dict] = []
    for index, spec in enumerate(MODEL_SPECS, start=1):
        print(
            f"Predicting {index}/{len(MODEL_SPECS)}: "
            f"{spec.model_id} ({spec.algorithm}, {spec.feature_count} features)",
            flush=True,
        )
        frame, model_metadata = _predict_model(spec, pairs, forward, reverse)
        nested, rows, class_rows, matrix_rows = _evaluate_predictions(
            spec, frame, masks
        )
        predictions[spec.model_id] = frame
        metadata[spec.model_id] = model_metadata
        metrics_by_model[spec.model_id] = nested
        comparison_rows.extend(rows)
        per_class_rows.extend(class_rows)
        confusion_rows.extend(matrix_rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(OUTPUT_DIR / "experimental_pairs.csv", index=False)
    pd.concat([pairs[["pair_id"]], forward[ALL_COMPUTED_FEATURE_COLUMNS]], axis=1).to_csv(
        OUTPUT_DIR / "experimental_pair_features.csv", index=False
    )
    for model_id in PRIMARY_MODEL_IDS:
        predictions[model_id].to_csv(
            OUTPUT_DIR / f"{model_id}_predictions.csv", index=False
        )

    primary_wide = _wide_predictions(pairs, predictions, PRIMARY_MODEL_IDS)
    primary_wide.to_csv(OUTPUT_DIR / "prediction_comparison.csv", index=False)
    sensitivity_ids = tuple(
        dict.fromkeys((*CATBOOST_CONTROL_IDS, *F27_PROTOCOL_IDS))
    )
    _wide_predictions(pairs, predictions, sensitivity_ids).to_csv(
        OUTPUT_DIR / "sensitivity_prediction_comparison.csv", index=False
    )

    comparison = pd.DataFrame(comparison_rows)
    primary_comparison = _comparison_subset(
        comparison, PRIMARY_MODEL_IDS, "primary_requested_models"
    )
    primary_comparison.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    sensitivity_comparison = pd.concat(
        [
            _comparison_subset(
                comparison,
                CATBOOST_CONTROL_IDS,
                "same_algorithm_catboost",
            ),
            _comparison_subset(
                comparison,
                F27_PROTOCOL_IDS,
                "f27_training_protocol_reference",
            ),
        ],
        ignore_index=True,
    )
    sensitivity_comparison.to_csv(
        OUTPUT_DIR / "sensitivity_model_comparison.csv", index=False
    )
    comparison.to_csv(OUTPUT_DIR / "all_model_metrics.csv", index=False)
    pd.DataFrame(per_class_rows).to_csv(
        OUTPUT_DIR / "per_class_metrics.csv", index=False
    )
    pd.DataFrame(confusion_rows).to_csv(
        OUTPUT_DIR / "confusion_matrices.csv", index=False
    )

    result = {
        "input": str(SUMMARY_PATH.relative_to(ROOT)),
        "output_directory": str(OUTPUT_DIR.relative_to(ROOT)),
        "pairing_policy": (
            "Adjacent rows only within each positional raw-file group segment; "
            "repeated group names are not merged."
        ),
        "label_policy": (
            "decrease if delta < -1e-4 S/cm; unchanged if "
            "abs(delta) <= 1e-4 S/cm; increase if delta > 1e-4 S/cm"
        ),
        "feature_limitations": [
            "Text after ' + ' is omitted from composition descriptors.",
            "Preparation method is reported but is not a model feature.",
            "Unseen family values become missing categorical values for LightGBM.",
        ],
        "data_audit": data_audit,
        "scope_pair_counts": {
            scope: int(mask.sum()) for scope, mask in masks.items()
        },
        "comparison_sets": {
            "primary_requested_models": list(PRIMARY_MODEL_IDS),
            "same_algorithm_catboost": list(CATBOOST_CONTROL_IDS),
            "f27_training_protocol_reference": list(F27_PROTOCOL_IDS),
        },
        "model_metadata": metadata,
        "metrics": metrics_by_model,
    }
    (OUTPUT_DIR / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    overall = primary_comparison[primary_comparison["scope"] == "overall"]
    print("\nPrimary experimental comparison:", flush=True)
    print(
        overall[
            [
                "model_id",
                "feature_count",
                "algorithm",
                "pairs",
                "accuracy",
                "macro_f1",
                "balanced_accuracy",
                "severe_reversal_rate",
            ]
        ].to_string(index=False),
        flush=True,
    )
    print(f"Outputs: {OUTPUT_DIR}", flush=True)
    return result


def main() -> None:
    """Run the point-in-time experimental model comparison."""
    compare_experimental_models()


if __name__ == "__main__":
    main()
