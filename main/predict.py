from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

import joblib
import numpy as np
import pandas as pd

from .features import SMALL_FEATURE_SPECS, TARGET_COLUMN, FeatureConfig, make_feature_table
from .paths import EXPERIMENTAL_ANNOTATIONS_DIR, portable_path


DEFAULT_EXPERIMENTAL_FAMILY_BLOCKS = EXPERIMENTAL_ANNOTATIONS_DIR / "family_blocks_legacy.csv"
DUMMY_CONDUCTIVITY = 1e-6


@dataclass
class PredictConfig:
    """Options for formula prediction from a saved training run.

    model_name:
        Subdirectory/model to load from the training output. Use None to read
        best_model.txt, falling back to lightgbm if that file is absent.
    output_dir:
        Optional exact output directory. When omitted, save predictions inside
        the selected training run.
    output_root:
        Optional alternate run root. The default keeps predictions under
        <model_output_dir>/predictions/<model>/<dataset>/.
    dataset_name:
        Optional dataset label. File inputs default to the input stem.
    prediction_purpose:
        Short description stored in run_metadata.json.
    formula_column:
        Input column containing formulas. If None, common names such as
        True Composition, formula, Formula, and composition are auto-detected.
    id_column:
        Optional input ID/name column. If None, sequential pred_0001 IDs are
        generated.
    family_column:
        Optional input Family column. If None, Family/family is auto-detected;
        otherwise missing family is encoded as unknown.
    """

    model_name: str | None = None
    output_dir: Path | None = None
    output_root: Path | None = None
    dataset_name: str | None = None
    prediction_purpose: str | None = None
    formula_column: str | None = None
    id_column: str | None = None
    family_column: str | None = None


@dataclass
class PredictionResult:
    predictions: pd.DataFrame
    features: pd.DataFrame
    output_dir: Path | None = None
    evaluation: dict | None = None


def _safe_path_name(value: str, fallback: str) -> str:
    """Return a stable, filesystem-safe name."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._-")
    return cleaned or fallback


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dataset_name(input_data, configured_name: str | None) -> str:
    if configured_name:
        return _safe_path_name(configured_name, "prediction_data")
    if isinstance(input_data, (str, Path)):
        return _safe_path_name(Path(input_data).stem, "prediction_data")
    return "inline_data"


def _prediction_output_dir(
    input_data,
    model_output_dir: Path,
    model_name: str,
    config: PredictConfig,
) -> Path:
    if config.output_dir is not None:
        return Path(config.output_dir)
    training_run = _safe_path_name(model_output_dir.name, "model_run")
    selected_model = _safe_path_name(model_name, "model")
    dataset = _dataset_name(input_data, config.dataset_name)
    if config.output_root is None:
        return model_output_dir / "predictions" / selected_model / dataset
    return Path(config.output_root) / training_run / "predictions" / selected_model / dataset


def _adjacent_trend_metrics(predictions: pd.DataFrame) -> dict:
    """Compare adjacent directions within each ordered composition series."""
    required = {
        "base_formula",
        "source_row",
        "true_log10_conductivity",
        "pred_log10_conductivity",
    }
    if not required.issubset(predictions.columns):
        return {
            "adjacent_trend_status": "not_available",
            "adjacent_trend_reason": "base_formula and source_row are required.",
        }

    frame = predictions[list(required)].copy()
    for column in ("source_row", "true_log10_conductivity", "pred_log10_conductivity"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna().sort_values("source_row")
    correct = 0
    comparable = 0
    for _, group in frame.groupby("base_formula", sort=False):
        if len(group) < 2:
            continue
        true_delta = np.diff(group["true_log10_conductivity"].to_numpy())
        predicted_delta = np.diff(group["pred_log10_conductivity"].to_numpy())
        mask = (true_delta != 0) & (predicted_delta != 0)
        comparable += int(np.sum(mask))
        correct += int(np.sum(np.sign(true_delta[mask]) == np.sign(predicted_delta[mask])))
    mismatches = comparable - correct
    return {
        "adjacent_trend_status": "ok" if comparable else "not_available",
        "n_adjacent_comparable": comparable,
        "n_adjacent_correct": correct,
        "n_adjacent_mismatches": mismatches,
        "adjacent_trend_accuracy": float(correct / comparable) if comparable else None,
    }


def label_families_by_blocks(
    input_path: str | Path,
    block_config: str | Path = DEFAULT_EXPERIMENTAL_FAMILY_BLOCKS,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Label ordered doping series with the family of their base material."""
    input_path = Path(input_path)
    rows = pd.read_csv(
        input_path,
        sep="\t",
        comment="#",
        header=None,
        names=["True Composition", "reference_mS_cm"],
        usecols=[0, 1],
        dtype={"True Composition": str},
    )
    rows["True Composition"] = rows["True Composition"].str.strip()
    rows["ID"] = [f"exp_{index:03d}" for index in range(1, len(rows) + 1)]
    rows["source_row"] = np.arange(1, len(rows) + 1)
    rows["base_formula"] = pd.NA
    rows["Family"] = pd.NA

    blocks = pd.read_csv(block_config)
    covered = np.zeros(len(rows), dtype=bool)
    for block in blocks.itertuples(index=False):
        start = int(block.start_row)
        end = int(block.end_row)
        if start < 1 or end < start or end > len(rows):
            raise ValueError(f"Invalid family block {start}-{end} for {len(rows)} rows")
        selection = rows["source_row"].between(start, end)
        if covered[selection.to_numpy()].any():
            raise ValueError(f"Overlapping family block {start}-{end}")
        rows.loc[selection, "base_formula"] = str(block.base_formula)
        rows.loc[selection, "Family"] = str(block.family)
        covered |= selection.to_numpy()

    if not covered.all():
        missing = rows.loc[~covered, "source_row"].tolist()
        raise ValueError(f"Family blocks do not cover source rows: {missing}")

    columns = ["ID", "True Composition", "reference_mS_cm", "base_formula", "Family", "source_row"]
    labeled = rows[columns]
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        labeled.to_csv(output_path, index=False)
    return labeled


def _looks_like_header(parts: list[str]) -> bool:
    candidates = {"true composition", "formula", "composition", "family", "id", "sample_id", "name"}
    normalized = {part.strip().lower() for part in parts}
    return bool(normalized & candidates)


def _read_delimited_input(path: Path, sep: str, first_line: str) -> pd.DataFrame:
    parts = [part.strip() for part in first_line.split(sep)]
    if _looks_like_header(parts):
        return pd.read_csv(path, sep=sep, dtype=str)
    column_count = max(len(parts), 1)
    columns = ["True Composition"]
    if column_count >= 2:
        columns.append("reference_value")
    columns.extend(f"extra_{index}" for index in range(3, column_count + 1))
    return pd.read_csv(path, sep=sep, dtype=str, header=None, names=columns, comment="#")


def _read_input(input_data, config: PredictConfig) -> pd.DataFrame:
    if isinstance(input_data, pd.DataFrame):
        frame = input_data.copy()
    elif isinstance(input_data, (list, tuple, pd.Series)):
        frame = pd.DataFrame({"True Composition": list(input_data)})
    else:
        path = Path(input_data)
        first_line = path.read_text(encoding="utf-8-sig").splitlines()[0]
        if "," in first_line or "\t" in first_line:
            sep = "\t" if "\t" in first_line and "," not in first_line else ","
            frame = _read_delimited_input(path, sep, first_line)
        else:
            frame = pd.DataFrame(
                {"True Composition": [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]}
            )

    columns = list(frame.columns)
    formula_column = config.formula_column or _find_column(columns, ("True Composition", "formula", "Formula", "composition"))
    if formula_column is None:
        if len(columns) == 1:
            formula_column = columns[0]
        else:
            raise ValueError("Could not determine formula column")
    id_column = config.id_column or _find_column(columns, ("ID", "id", "sample_id", "name"))
    family_column = config.family_column or _find_column(columns, ("Family", "family"))

    output = pd.DataFrame()
    output["ID"] = frame[id_column].astype(str) if id_column else [f"pred_{index:04d}" for index in range(1, len(frame) + 1)]
    output["True Composition"] = frame[formula_column].astype(str).str.strip()
    output[TARGET_COLUMN] = DUMMY_CONDUCTIVITY
    if family_column is not None:
        output["Family"] = frame[family_column].astype(str).str.strip()
    for column in ("base_formula", "reference_mS_cm", "source_row"):
        if column in frame.columns:
            output[column] = frame[column].to_numpy()
    return output


def _find_column(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    normalized = {column.strip().lower(): column for column in columns}
    for candidate in candidates:
        match = normalized.get(candidate.lower())
        if match is not None:
            return match
    return None


def _model_dir(output_dir: Path, model_name: str | None) -> tuple[Path, str]:
    if model_name is None:
        best_path = output_dir / "best_model.txt"
        if best_path.exists():
            model_name = best_path.read_text(encoding="utf-8").strip()
        else:
            model_name = "lightgbm"
    return output_dir / model_name, model_name


def predict_formulas(
    input_data,
    model_output_dir: str | Path,
    config: PredictConfig | None = None,
) -> PredictionResult:
    """Generate descriptors for formulas and predict conductivity with a saved model."""

    config = config or PredictConfig()
    model_output_dir = Path(model_output_dir)
    model_dir, model_name = _model_dir(model_output_dir, config.model_name)
    artifact = joblib.load(model_dir / "model.joblib")
    model = artifact["model"]
    scaler = artifact.get("scaler")
    feature_columns = list(artifact["feature_cols"])
    medians = pd.Series(artifact.get("feature_medians", 0.0))

    family_mapping = artifact.get("family_mapping")
    summary_path = model_output_dir / "summary.json"
    if family_mapping is None and summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        family_mapping = summary.get("family_mapping")

    formulas = _read_input(input_data, config)
    feature_result = make_feature_table(
        formulas,
        FeatureConfig(
            min_conductivity=None,
            include_family="family" in feature_columns,
            include_interactions=True,
            include_small_features=any(
                column in feature_columns
                for column, *_ in SMALL_FEATURE_SPECS
            ),
            family_mapping=family_mapping,
            output_path=None,
        ),
    )
    features = feature_result.table
    X = features.reindex(columns=feature_columns).apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(medians.reindex(feature_columns).fillna(0.0))
    predict_X = X
    if scaler is not None:
        predict_X = pd.DataFrame(scaler.transform(X), columns=feature_columns, index=X.index)
    pred_log10 = model.predict(predict_X)

    metadata_columns = [
        column
        for column in ("ID", "True Composition", "base_formula", "Family", "reference_mS_cm", "source_row")
        if column in features.columns
    ]
    predictions = features[metadata_columns].copy()
    predictions["model_name"] = model_name
    predictions["pred_log10_conductivity"] = pred_log10
    predictions["pred_conductivity_S_cm-1"] = np.power(10.0, pred_log10)
    evaluation = {
        "status": "not_available",
        "reason": "Input data does not contain reference_mS_cm values.",
        "rows_total": int(len(predictions)),
        "rows_evaluated": 0,
    }
    if "reference_mS_cm" in predictions.columns:
        evaluation["reason"] = "No valid positive reference_mS_cm values were found."
        reference = pd.to_numeric(predictions["reference_mS_cm"], errors="coerce") / 1000.0
        predictions["true_conductivity_S_cm-1"] = reference
        predictions["true_log10_conductivity"] = np.log10(reference.where(reference > 0))
        predictions["residual_log10"] = (
            predictions["true_log10_conductivity"] - predictions["pred_log10_conductivity"]
        )
        valid = predictions[["true_log10_conductivity", "pred_log10_conductivity"]].dropna()
        if not valid.empty:
            residual = valid["true_log10_conductivity"] - valid["pred_log10_conductivity"]
            true_values = valid["true_log10_conductivity"].to_numpy()
            predicted_values = valid["pred_log10_conductivity"].to_numpy()
            comparable_pairs = 0
            inverted_pairs = 0
            predicted_tie_pairs = 0
            for left in range(len(valid)):
                for right in range(left + 1, len(valid)):
                    true_direction = np.sign(true_values[left] - true_values[right])
                    if true_direction == 0:
                        continue
                    comparable_pairs += 1
                    predicted_direction = np.sign(predicted_values[left] - predicted_values[right])
                    if predicted_direction == 0:
                        predicted_tie_pairs += 1
                    elif predicted_direction != true_direction:
                        inverted_pairs += 1
            denominator = np.sum(
                (valid["true_log10_conductivity"] - valid["true_log10_conductivity"].mean()) ** 2
            )
            evaluation = {
                "status": "ok",
                "rows_total": int(len(predictions)),
                "rows_evaluated": int(len(valid)),
                "mae_log10": float(np.mean(np.abs(residual))),
                "rmse_log10": float(np.sqrt(np.mean(residual**2))),
                "r2_log10": float(1.0 - np.sum(residual**2) / denominator) if denominator > 0 else None,
                "spearman_rank_correlation": float(
                    valid["true_log10_conductivity"].corr(
                        valid["pred_log10_conductivity"], method="spearman"
                    )
                ),
                "kendall_rank_correlation": float(
                    valid["true_log10_conductivity"].corr(
                        valid["pred_log10_conductivity"], method="kendall"
                    )
                ),
                "n_comparable_pairs": comparable_pairs,
                "n_inverted_pairs": inverted_pairs,
                "n_predicted_tie_pairs": predicted_tie_pairs,
                "pairwise_inversion_rate": (
                    float(inverted_pairs / comparable_pairs) if comparable_pairs else None
                ),
            }
            evaluation.update(_adjacent_trend_metrics(predictions))

    output_dir = _prediction_output_dir(input_data, model_output_dir, model_name, config)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    features.to_csv(output_dir / "features.csv", index=False)
    (output_dir / "evaluation.json").write_text(
        json.dumps(evaluation, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    input_path = Path(input_data).resolve() if isinstance(input_data, (str, Path)) else None
    model_path = (model_dir / "model.joblib").resolve()
    metadata = {
        "training_run": model_output_dir.name,
        "model_name": model_name,
        "model_output_dir": portable_path(model_output_dir),
        "dataset_name": _dataset_name(input_data, config.dataset_name),
        "input_path": portable_path(input_path) if input_path is not None else None,
        "input_sha256": _sha256(input_path) if input_path is not None else None,
        "model_path": portable_path(model_path),
        "model_sha256": _sha256(model_path),
        "prediction_purpose": config.prediction_purpose or "Not specified",
        "rows": int(len(predictions)),
        "files": {
            "predictions": "predictions.csv",
            "features": "features.csv",
            "evaluation": "evaluation.json",
        },
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return PredictionResult(
        predictions=predictions,
        features=features,
        output_dir=output_dir,
        evaluation=evaluation,
    )
