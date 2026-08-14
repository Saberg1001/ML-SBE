"""Reusable trend-classification prediction (build pairs -> classify).

Merges the former one-off ``experimental_predict_pairs.py`` (group the manually
grouped experimental file into adjacent pairs) and ``experimental_predict_trend.py``
(classify each pair as decrease / unchanged / increase) into one runnable entry
point with no argparse interaction.

Run directly via ``python main/trend/predict.py`` (the VS Code "Run" button):

Input   : data/experimental/raw/experimental-data.csv        (grouped experimental table)
          data/experimental/annotations/experimental-data-labeled.csv (source_row -> Family)
          runs/trend/<run>/catboost/model.joblib             (trained trend classifier)
Outputs : data/experimental/annotations/experimental-data-predict-trend.csv
          data/experimental/annotations/experimental-data-predict-trend-metrics.json

All parameters come from ``main.trend.pipeline.default_trend_predict_config``.
"""

from __future__ import annotations

import os
import sys

# Allow running this module directly as a script (e.g. VS Code "Run" button):
# project root must be importable for ``main.*``.
if __package__ is None:
    _PROJECT_ROOT = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

import json
from pathlib import Path
import re

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score

from main.trend.features import (
    MODEL_FEATURE_COLUMNS,
    _formula_descriptor_cache,
    _pair_numeric_features,
    classify_trend_delta,
)
from main.trend.pipeline import TrendPredictConfig, default_trend_predict_config


def _parse_conductivity(text: str, scale: float) -> float:
    value = text.split("#", 1)[0].strip()
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*x\s*10\s*([+-]?[0-9]+)", value, re.I)
    if match:
        return float(match.group(1)) * (10.0 ** int(match.group(2))) * scale
    return float(value) * scale


def _parse_formula(formula: str) -> str:
    # Additive descriptions such as ``+ 5wt%ZrCl4`` are not part of the
    # host composition descriptor; retain the original text in the output.
    return formula.split(" + ", 1)[0].strip()


def _grouped_raw_rows(raw_path: Path) -> list[dict[str, str]]:
    """Read grouped experimental rows with optional leading IDs."""
    rows: list[dict[str, str]] = []
    current_group = ""
    group_segment = 0
    for line in raw_path.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) == 1:
            if parts[0].strip():
                current_group = parts[0].strip()
                group_segment += 1
            continue
        if len(parts) >= 3:
            row_id, formula, conductivity = (
                parts[0].strip(), parts[1].strip(), parts[2].strip()
            )
        else:
            row_id = ""
            formula, conductivity = parts[0].strip(), parts[1].strip()
        if not current_group or not formula or not conductivity:
            continue
        try:
            _parse_conductivity(conductivity, 1.0)
        except ValueError:
            continue
        rows.append({
            "id": row_id,
            "formula": formula,
            "conductivity": conductivity,
            "group_id": current_group,
            "group_segment": str(group_segment),
            "source_row": str(len(rows) + 1),
        })
    return rows


def _build_pairs(raw_path: Path, annotation_path: Path) -> pd.DataFrame:
    """Parse the manually grouped raw file into adjacent composition pairs."""
    annotations = pd.read_csv(annotation_path, dtype=str, keep_default_na=False)
    family_by_source = dict(zip(annotations["source_row"], annotations["Family"]))
    family_by_id = dict(zip(annotations["ID"], annotations["Family"]))
    formula_by_id = dict(zip(annotations["ID"], annotations["True Composition"]))

    grouped: dict[str, list[dict[str, str]]] = {}
    for row in _grouped_raw_rows(raw_path):
        row_id = row["id"]
        if row_id and row_id in formula_by_id:
            if formula_by_id[row_id] != row["formula"]:
                raise ValueError(
                    f"Formula mismatch for {row_id}: raw={row['formula']!r}, "
                    f"annotation={formula_by_id[row_id]!r}"
                )
            family = family_by_id[row_id]
        else:
            family = family_by_source.get(row["source_row"], "unknown")
        # Repeated display labels can denote separate blocks in the raw file.
        grouped.setdefault(row["group_segment"], []).append({
            **row,
            "family": family,
        })

    records: list[dict[str, str]] = []
    pair_number = 0
    for rows in grouped.values():
        group_id = rows[0]["group_id"]
        for left, right in zip(rows, rows[1:]):
            pair_number += 1
            records.append({
                "group_id": group_id,
                "pair_id": f"exp_pair_{pair_number:04d}",
                "id_a": left["id"],
                "id_b": right["id"],
                "化学式_a": left["formula"],
                "化学式_b": right["formula"],
                "family": left["family"] if left["family"] == right["family"] else "mixed",
                "source_row_a": left["source_row"],
                "source_row_b": right["source_row"],
            })
    return pd.DataFrame(records)


def _row_conductivity(raw_path: Path, raw_scale: float) -> dict[str, float]:
    """Map 1-based source_row -> conductivity parsed from the raw tab table."""
    return {
        row["source_row"]: _parse_conductivity(row["conductivity"], raw_scale)
        for row in _grouped_raw_rows(raw_path)
    }


def _prediction_matrix(
    model_input: pd.DataFrame,
    family: pd.Series,
    numeric_features: list[str],
    model_name: str,
    preprocessing: dict,
    medians: pd.Series,
):
    numeric = model_input[numeric_features].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.fillna(medians.reindex(numeric_features).fillna(0.0))
    if model_name in {"random_forest", "xgboost"} or preprocessing.get("encoded_feature_names"):
        categories = list(preprocessing["family_categories"])
        family_codes = pd.Categorical(family, categories=categories)
        one_hot = sparse.csr_matrix(
            pd.get_dummies(family_codes, dtype=np.float32).to_numpy()
        )
        numeric_matrix = sparse.csr_matrix(numeric.to_numpy(dtype=np.float32))
        return sparse.hstack([numeric_matrix, one_hot], format="csr")
    result = numeric.copy()
    if model_name == "lightgbm":
        result["family"] = pd.Categorical(
            family, categories=preprocessing.get("family_categories", [])
        )
    else:
        result["family"] = family.to_numpy()
    return result


def predict_trend(config: TrendPredictConfig | None = None) -> dict:
    """Build pairs from the raw file, classify each pair, write CSV + metrics.

    Returns the metrics dict. All I/O paths come from ``config`` (defaults from
    :func:`default_trend_predict_config`).
    """
    config = config or default_trend_predict_config()
    pairs = _build_pairs(config.raw_csv, config.annotations)
    if pairs.empty:
        raise ValueError(f"No pairs built from {config.raw_csv.resolve()}")

    row_values = _row_conductivity(config.raw_csv, config.raw_scale)
    formula_a = pairs["化学式_a"].map(_parse_formula)
    formula_b = pairs["化学式_b"].map(_parse_formula)
    formulas = pd.unique(pd.concat([formula_a, formula_b], ignore_index=True))
    cache = _formula_descriptor_cache(formulas, show_progress=True)
    feature_rows = [
        _pair_numeric_features(cache[fa], cache[fb], fa, fb)
        for fa, fb in zip(formula_a, formula_b)
    ]
    numeric = pd.DataFrame(feature_rows)
    sigma_a = pairs["source_row_a"].map(lambda x: row_values[x]).to_numpy(float)
    sigma_b = pairs["source_row_b"].map(lambda x: row_values[x]).to_numpy(float)

    model_bundle = joblib.load(config.model)
    model = model_bundle["model"]
    model_input = numeric.copy()
    numeric_features = list(model_bundle.get("numeric_features", MODEL_FEATURE_COLUMNS))
    if "conductivity_a_S_cm-1" in numeric_features:
        model_input["conductivity_a_S_cm-1"] = sigma_a
    family = pairs["family"].astype(str)
    model_name = str(model_bundle.get("model_name", "")).lower()
    preprocessing = model_bundle.get("preprocessing", {})
    if not preprocessing:
        preprocessing_path = config.model.with_name("preprocessing.json")
        if preprocessing_path.exists():
            preprocessing = json.loads(preprocessing_path.read_text(encoding="utf-8"))
    medians = pd.Series(
        model_bundle.get("numeric_medians", preprocessing.get("numeric_medians", {})),
        dtype=float,
    )
    prediction_input = _prediction_matrix(
        model_input, family, numeric_features, model_name, preprocessing, medians
    )
    forward_probabilities = model.predict_proba(prediction_input)
    if model_bundle.get("swap_symmetric_prediction", False):
        reverse_rows = [
            _pair_numeric_features(cache[fb], cache[fa], fb, fa)
            for fa, fb in zip(formula_a, formula_b)
        ]
        reverse_input = pd.DataFrame(reverse_rows)
        if "conductivity_a_S_cm-1" in numeric_features:
            reverse_input["conductivity_a_S_cm-1"] = sigma_b
        reverse_matrix = _prediction_matrix(
            reverse_input, family, numeric_features, model_name, preprocessing, medians
        )
        reverse_aligned = model.predict_proba(reverse_matrix)[:, [2, 1, 0]]
        probabilities = (forward_probabilities + reverse_aligned) / 2.0
    else:
        reverse_aligned = forward_probabilities
        probabilities = forward_probabilities
    labels = list(config.class_labels)
    predicted = np.array([labels[i] for i in probabilities.argmax(axis=1)])

    delta = sigma_b - sigma_a
    threshold = config.threshold_s_cm
    true = classify_trend_delta(delta, threshold_s_cm=threshold)

    output = pairs.copy()
    output["电导率_a_S_cm-1"] = sigma_a
    output["电导率_b_S_cm-1"] = sigma_b
    output["真实电导率变化值_S_cm-1"] = delta
    output["真实趋势标签"] = true
    output["预测趋势标签"] = predicted
    output["预测正确"] = true == predicted
    for i, label in enumerate(labels):
        output[f"probability_{label}"] = probabilities[:, i]
    output_path = Path(config.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)

    metrics = {
        "model": str(config.model),
        "pairs": len(output),
        "threshold_S_cm-1": threshold,
        "accuracy": float(accuracy_score(true, predicted)),
        "macro_f1": float(f1_score(true, predicted, labels=labels, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(true, predicted)),
        "labels": {label: int((true == label).sum()) for label in labels},
        "predicted_labels": {label: int((predicted == label).sum()) for label in labels},
        "confusion_matrix_labels": labels,
        "confusion_matrix": confusion_matrix(true, predicted, labels=labels).tolist(),
    }
    metrics_path = Path(config.metrics)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metrics


def main() -> None:
    """Run trend prediction with the default configuration (point-run entry).

    Run directly via ``python main/trend/predict.py`` (or the VS Code "Run"
    button). Uses :func:`default_trend_predict_config` for inputs/model/output.
    Writes :data:`experimental-data-predict-trend.csv` and the matching
    ``-metrics.json`` under ``data/experimental/annotations/``.
    """
    from main.trend.pipeline import default_trend_predict_config

    config = default_trend_predict_config()
    metrics = predict_trend(config)
    print(f"Predicted/dT pairs : {metrics['pairs']}")
    print(f"Accuracy           : {metrics['accuracy']:.4f}")
    print(f"Prediction CSV     : {Path(config.output).resolve()}")
    print(f"Metrics JSON       : {Path(config.metrics).resolve()}")


if __name__ == "__main__":
    main()
