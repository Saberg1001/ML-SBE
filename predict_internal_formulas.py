from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from pymatgen.core import Composition


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "rawdata" / "expriment-test"
DEFAULT_MODEL_DIR = ROOT / "models" / "outputs_optuna" / "ionic_26_features_random_gt1e-6_50"
DEFAULT_OUTPUT = ROOT / "predictions" / "expriment-test_predictions.csv"
TARGET_COLUMN = "Ionic conductivity (S cm-1)"
DUMMY_CONDUCTIVITY = 1e-6

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from get_feature.get_feature import (  # noqa: E402
    charge_residual,
    composition_features,
    contains_organic_molecule,
    oxidation_state_guesses,
)
from models.feature_engineering import engineer_features  # noqa: E402
from models.feature_engineering import parse_conductivity  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate features for formulas and predict ionic conductivity."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=None,
        help="Optional path for ranking metrics. Defaults to *_metrics.json.",
    )
    parser.add_argument(
        "--formula-column",
        default=None,
        help="Formula column name for CSV/TSV input. Auto-detected by default.",
    )
    parser.add_argument(
        "--id-column",
        default=None,
        help="ID column name for CSV/TSV input. Auto-detected by default.",
    )
    parser.add_argument(
        "--conductivity-column",
        default=None,
        help="Conductivity column name for CSV/TSV input. Auto-detected by default.",
    )
    parser.add_argument(
        "--true-conductivity-scale",
        type=float,
        default=0.01,
        help="Multiplier applied to input conductivity before comparing with S/cm predictions.",
    )
    parser.add_argument(
        "--exclude-elements-for-metrics",
        nargs="*",
        default=[],
        help="Element symbols excluded from ranking metrics only.",
    )
    parser.add_argument(
        "--features-output",
        type=Path,
        default=None,
        help="Optional path for the generated feature table.",
    )
    parser.add_argument(
        "--allow-organic",
        action="store_true",
        help="Predict formulas containing both C and H instead of skipping them.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce progress output.",
    )
    return parser.parse_args()


def first_nonempty_line(path: Path) -> str:
    with path.open("r", encoding="utf-8-sig") as file:
        for line in file:
            text = line.strip()
            if text:
                return text
    return ""


def find_column(columns: list[str], explicit: str | None, candidates: tuple[str, ...]) -> str | None:
    if explicit:
        if explicit not in columns:
            raise ValueError(f"Column not found: {explicit}")
        return explicit

    normalized = {col.strip().lower(): col for col in columns}
    for candidate in candidates:
        match = normalized.get(candidate.lower())
        if match is not None:
            return match
    return None


def looks_like_header(cells: list[object]) -> bool:
    header_names = {
        "id",
        "sample_id",
        "name",
        "formula",
        "composition",
        "true composition",
        "conductivity",
        TARGET_COLUMN.lower(),
    }
    normalized = {str(cell).strip().lower() for cell in cells if not pd.isna(cell)}
    return bool(normalized & header_names)


def read_formula_input(
    path: Path,
    *,
    formula_column: str | None = None,
    id_column: str | None = None,
    conductivity_column: str | None = None,
    true_conductivity_scale: float = 1.0,
) -> pd.DataFrame:
    first_line = first_nonempty_line(path)
    if not first_line:
        raise ValueError(f"Input file is empty: {path}")

    if "," in first_line or "\t" in first_line:
        sep = "\t" if "\t" in first_line and "," not in first_line else ","
        preview = pd.read_csv(path, sep=sep, header=None, dtype=str, nrows=1)
        has_header = looks_like_header(preview.iloc[0].tolist())
        if has_header:
            frame = pd.read_csv(path, sep=sep, dtype=str)
            formula_col = find_column(
                list(frame.columns),
                formula_column,
                ("True Composition", "formula", "Formula", "composition", "Composition"),
            )
            if formula_col is None:
                if len(frame.columns) == 1:
                    formula_col = frame.columns[0]
                else:
                    raise ValueError(
                        "Could not detect the formula column. Use --formula-column."
                    )
            id_col = find_column(list(frame.columns), id_column, ("ID", "id", "sample_id", "name"))
            conductivity_col = find_column(
                list(frame.columns),
                conductivity_column,
                (
                    TARGET_COLUMN,
                    "conductivity",
                    "Conductivity",
                    "ionic conductivity",
                    "Ionic conductivity",
                ),
            )
        else:
            frame = pd.read_csv(path, sep=sep, header=None, dtype=str)
            formula_col = frame.columns[0]
            id_col = None
            conductivity_col = frame.columns[1] if len(frame.columns) > 1 else None
            if formula_column is not None or id_column is not None or conductivity_column is not None:
                raise ValueError(
                    "Column names can only be used when the delimited input has a header row."
                )

        output_data = {
            "ID": frame[id_col].astype(str) if id_col is not None else make_ids(len(frame)),
            "True Composition": frame[formula_col].astype(str).str.strip(),
        }
        if conductivity_col is not None:
            output_data["true_conductivity"] = frame[conductivity_col].astype(str).str.strip()
        output = pd.DataFrame(output_data)
    else:
        with path.open("r", encoding="utf-8-sig") as file:
            formulas = [line.strip() for line in file if line.strip()]
        output = pd.DataFrame(
            {
                "ID": make_ids(len(formulas)),
                "True Composition": formulas,
            }
        )

    output = output[output["True Composition"].astype(str).str.len() > 0].reset_index(drop=True)
    if "true_conductivity" in output.columns:
        parsed = output["true_conductivity"].apply(parse_conductivity)
        output["true_conductivity_scale"] = true_conductivity_scale
        output["true_conductivity_value"] = [
            item[0] * true_conductivity_scale if not math.isnan(item[0]) else math.nan
            for item in parsed
        ]
        output["true_conductivity_qualifier"] = [item[1] for item in parsed]
        output["true_log10_conductivity"] = np.log10(
            output["true_conductivity_value"].where(output["true_conductivity_value"] > 0)
        )
    if output.empty:
        raise ValueError(f"No formulas found in input file: {path}")
    return output


def make_ids(count: int) -> list[str]:
    width = max(3, len(str(count)))
    return [f"exp_{index:0{width}d}" for index in range(1, count + 1)]


def charge_balance_info(formula: str) -> tuple[float, str]:
    composition = Composition(formula)
    amounts = {
        elem.symbol: float(composition.get_el_amt_dict()[elem.symbol])
        for elem in composition.elements
    }
    guesses, _, _, _ = oxidation_state_guesses(composition)
    guess = guesses[0] if guesses else {}
    residual = charge_residual(amounts, guess)
    if abs(residual) >= 1.0:
        return residual, f"charge_residual={residual:.6g}"
    return residual, ""


def build_base_features(
    formulas: pd.DataFrame,
    *,
    allow_organic: bool,
    quiet: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    statuses: list[dict[str, object]] = []
    total = len(formulas)
    start = time.time()

    for index, row in formulas.iterrows():
        formula_id = str(row["ID"])
        formula = str(row["True Composition"]).strip()
        status = {
            "ID": formula_id,
            "True Composition": formula,
            "true_conductivity": row.get("true_conductivity", ""),
            "true_conductivity_scale": row.get("true_conductivity_scale", math.nan),
            "true_conductivity_value": row.get("true_conductivity_value", math.nan),
            "true_conductivity_qualifier": row.get("true_conductivity_qualifier", ""),
            "true_log10_conductivity": row.get("true_log10_conductivity", math.nan),
            "status": "ok",
            "message": "",
            "charge_residual": math.nan,
        }

        try:
            composition = Composition(formula)
            if contains_organic_molecule(composition) and not allow_organic:
                status["status"] = "skipped"
                status["message"] = "organic-like formula; pass --allow-organic to predict"
            else:
                residual, message = charge_balance_info(formula)
                status["charge_residual"] = residual
                if message:
                    status["message"] = message
                feature_values = composition_features(formula).to_dict()
                rows.append(
                    {
                        "ID": formula_id,
                        "True Composition": formula,
                        TARGET_COLUMN: DUMMY_CONDUCTIVITY,
                        **feature_values,
                    }
                )
        except Exception as exc:
            status["status"] = "error"
            status["message"] = str(exc)

        statuses.append(status)
        done = index + 1
        if not quiet and (done == 1 or done % 10 == 0 or done == total):
            elapsed = time.time() - start
            seconds_per_row = elapsed / done
            eta = max(total - done, 0) * seconds_per_row
            print(
                f"Processed formulas {done}/{total}; elapsed {elapsed:.1f}s; ETA {eta:.1f}s",
                flush=True,
            )

    return pd.DataFrame(rows), pd.DataFrame(statuses)


def load_model_artifact(model_dir: Path) -> tuple[object, object | None, list[str]]:
    artifact_path = model_dir / "lightgbm" / "model.joblib"
    if not artifact_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {artifact_path}")

    artifact = joblib.load(artifact_path)
    if isinstance(artifact, dict):
        model = artifact.get("model")
        scaler = artifact.get("scaler")
        feature_cols = artifact.get("feature_cols")
    else:
        model = artifact
        scaler = None
        feature_cols = None

    if model is None:
        raise ValueError(f"No model found in artifact: {artifact_path}")
    if not feature_cols:
        feature_list_path = model_dir / "data" / "feature_list.txt"
        feature_cols = [
            line.strip()
            for line in feature_list_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    return model, scaler, list(feature_cols)


def training_medians(model_dir: Path, feature_cols: list[str]) -> pd.Series:
    train_path = model_dir / "data" / "X_train.csv"
    if not train_path.exists():
        return pd.Series(0.0, index=feature_cols)
    train_x = pd.read_csv(train_path)
    train_x = train_x.reindex(columns=feature_cols)
    medians = train_x.apply(pd.to_numeric, errors="coerce").median()
    return medians.fillna(0.0)


def make_prediction_frame(
    base_features: pd.DataFrame,
    model_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model, scaler, feature_cols = load_model_artifact(model_dir)
    processed, _ = engineer_features(base_features, add_interactions=True)

    for column in feature_cols:
        if column not in processed.columns:
            processed[column] = np.nan

    x = processed.reindex(columns=feature_cols).apply(pd.to_numeric, errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan)
    missing_count = x.isna().sum(axis=1)
    x = x.fillna(training_medians(model_dir, feature_cols))

    predict_x = x
    if scaler is not None:
        predict_x = pd.DataFrame(
            scaler.transform(x),
            columns=feature_cols,
            index=x.index,
        )

    pred_log10 = model.predict(predict_x)
    predictions = processed[["ID", "True Composition"]].copy()
    predictions["pred_log10_conductivity"] = pred_log10
    predictions["pred_conductivity_S_cm-1"] = np.power(10.0, pred_log10)
    predictions["n_missing_features_filled"] = missing_count.to_numpy()
    feature_table = pd.concat(
        [
            processed[["ID", "True Composition"]],
            x.add_prefix("feature__"),
        ],
        axis=1,
    )
    return predictions, feature_table


def default_features_output(output: Path) -> Path:
    return output.with_name(f"{output.stem}_features{output.suffix}")


def default_metrics_output(output: Path) -> Path:
    return output.with_name(f"{output.stem}_metrics.json")


def add_rank_columns(result: pd.DataFrame) -> pd.DataFrame:
    result = result.copy()
    result["log10_error_pred_minus_true"] = np.nan
    result["abs_log10_error"] = np.nan
    result["true_rank_desc"] = np.nan
    result["pred_rank_desc"] = np.nan
    result["rank_error"] = np.nan

    true_values = pd.to_numeric(result.get("true_log10_conductivity"), errors="coerce")
    pred_values = pd.to_numeric(result.get("pred_log10_conductivity"), errors="coerce")
    valid = true_values.notna() & pred_values.notna()
    if valid.any():
        result.loc[valid, "log10_error_pred_minus_true"] = pred_values[valid] - true_values[valid]
        result.loc[valid, "abs_log10_error"] = result.loc[
            valid,
            "log10_error_pred_minus_true",
        ].abs()
        result.loc[valid, "true_rank_desc"] = true_values[valid].rank(
            ascending=False,
            method="min",
        )
        result.loc[valid, "pred_rank_desc"] = pred_values[valid].rank(
            ascending=False,
            method="min",
        )
        result.loc[valid, "rank_error"] = (
            result.loc[valid, "pred_rank_desc"] - result.loc[valid, "true_rank_desc"]
        )
    return result


def inversion_metrics(result: pd.DataFrame) -> dict[str, object]:
    true_values = pd.to_numeric(result.get("true_log10_conductivity"), errors="coerce")
    pred_values = pd.to_numeric(result.get("pred_log10_conductivity"), errors="coerce")
    valid = true_values.notna() & pred_values.notna()
    true_scores = true_values[valid].to_numpy()
    pred_scores = pred_values[valid].to_numpy()
    errors = pred_values[valid] - true_values[valid]

    discordant = 0
    comparable = 0
    pred_ties = 0
    n_items = len(true_scores)
    for i in range(n_items):
        for j in range(i + 1, n_items):
            true_delta = true_scores[i] - true_scores[j]
            pred_delta = pred_scores[i] - pred_scores[j]
            if true_delta == 0:
                continue
            comparable += 1
            if pred_delta == 0:
                pred_ties += 1
            elif true_delta * pred_delta < 0:
                discordant += 1

    inversion_rate = math.nan
    if comparable:
        inversion_rate = discordant / comparable

    return {
        "metric": "pairwise_inversion_rate",
        "scale": "log10 conductivity",
        "n_items": int(n_items),
        "n_comparable_pairs": int(comparable),
        "n_inverted_pairs": int(discordant),
        "n_predicted_tie_pairs": int(pred_ties),
        "inversion_rate": inversion_rate,
        "mean_log10_error_pred_minus_true": float(errors.mean()) if len(errors) else math.nan,
        "mean_abs_log10_error": float(errors.abs().mean()) if len(errors) else math.nan,
        "median_abs_log10_error": float(errors.abs().median()) if len(errors) else math.nan,
    }


def has_any_element(formula: object, elements: set[str]) -> bool:
    if not elements:
        return False
    try:
        composition = Composition(str(formula))
    except Exception:
        return False
    formula_elements = {element.symbol for element in composition.elements}
    return bool(formula_elements & elements)


def filter_metrics_rows(result: pd.DataFrame, exclude_elements: list[str]) -> tuple[pd.DataFrame, dict[str, object]]:
    elements = {element.strip() for element in exclude_elements if element.strip()}
    if not elements:
        return result, {"excluded_elements": [], "n_excluded_rows": 0}

    excluded = result["True Composition"].apply(lambda formula: has_any_element(formula, elements))
    return (
        result.loc[~excluded].copy(),
        {
            "excluded_elements": sorted(elements),
            "n_excluded_rows": int(excluded.sum()),
        },
    )


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    model_dir = args.model_dir.resolve()
    output_path = args.output.resolve()
    features_output = (
        args.features_output.resolve()
        if args.features_output is not None
        else default_features_output(output_path)
    )
    metrics_output = (
        args.metrics_output.resolve()
        if args.metrics_output is not None
        else default_metrics_output(output_path)
    )

    formulas = read_formula_input(
        input_path,
        formula_column=args.formula_column,
        id_column=args.id_column,
        conductivity_column=args.conductivity_column,
        true_conductivity_scale=args.true_conductivity_scale,
    )
    if not args.quiet:
        print(f"Input formulas: {len(formulas)}")
        print(f"Model dir: {model_dir}")

    base_features, statuses = build_base_features(
        formulas,
        allow_organic=args.allow_organic,
        quiet=args.quiet,
    )

    if base_features.empty:
        result = statuses.copy()
        result["pred_log10_conductivity"] = np.nan
        result["pred_conductivity_S_cm-1"] = np.nan
        result["n_missing_features_filled"] = np.nan
    else:
        predictions, feature_table = make_prediction_frame(base_features, model_dir)
        result = statuses.merge(predictions, on=["ID", "True Composition"], how="left")
        features_output.parent.mkdir(parents=True, exist_ok=True)
        feature_table.to_csv(features_output, index=False)
        if not args.quiet:
            print(f"Wrote generated features: {features_output}")

    result = add_rank_columns(result)
    metrics_frame, metrics_filter = filter_metrics_rows(
        result,
        args.exclude_elements_for_metrics,
    )
    metrics = inversion_metrics(metrics_frame)
    metrics.update(metrics_filter)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    metrics_output.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if not args.quiet:
        ok_count = int((result["status"] == "ok").sum())
        print(f"Wrote predictions: {output_path}")
        print(f"Wrote ranking metrics: {metrics_output}")
        print(f"Predicted rows: {ok_count}/{len(result)}")
        if not math.isnan(metrics["inversion_rate"]):
            print(
                "Inversion rate: "
                f"{metrics['inversion_rate']:.6f} "
                f"({metrics['n_inverted_pairs']}/{metrics['n_comparable_pairs']} pairs)"
            )
        if metrics["excluded_elements"]:
            print(
                "Excluded from metrics: "
                f"{', '.join(metrics['excluded_elements'])} "
                f"({metrics['n_excluded_rows']} rows)"
            )


if __name__ == "__main__":
    main()
