from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from train_lightgbm import DEFAULT_TRAIN, TARGET_COLUMN, add_target_columns


ROOT = Path(__file__).resolve().parent
DEFAULT_PREDICTIONS = ROOT / "outputs" / "lightgbm" / "test_predictions.csv"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "lightgbm" / "error_analysis"
ANION_ELEMENTS = ("O", "S", "Se", "Te", "F", "Cl", "Br", "I")
ELEMENT_PATTERN = re.compile(r"([A-Z][a-z]?)")


def infer_anion_system(composition: object) -> str:
    elements = set(ELEMENT_PATTERN.findall(str(composition)))
    anions = [element for element in ANION_ELEMENTS if element in elements]
    return "+".join(anions) if anions else "other"


def summarize_errors(predictions: pd.DataFrame) -> dict[str, object]:
    abs_error = predictions["abs_log10_error"]
    return {
        "rows": int(len(predictions)),
        "rmse_log10": float(np.sqrt(np.mean(np.square(predictions["log10_error"])))),
        "mae_log10": float(abs_error.mean()),
        "median_abs_log10_error": float(abs_error.median()),
        "p90_abs_log10_error": float(abs_error.quantile(0.9)),
        "max_abs_log10_error": float(abs_error.max()),
        "abs_error_gt_1_count": int((abs_error > 1).sum()),
        "abs_error_gt_2_count": int((abs_error > 2).sum()),
        "upper_bound_test_count": int((predictions["conductivity_qualifier"] == "upper_bound").sum()),
    }


def build_duplicate_summary(train_path: Path, predictions: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    train = pd.read_csv(train_path)
    train = add_target_columns(
        train,
        upper_bound_threshold=args.upper_bound_threshold,
        upper_bound_replacement=args.upper_bound_replacement,
    )
    train_summary = (
        train.groupby("True Composition")["log10_conductivity"]
        .agg(train_count="count", train_log10_min="min", train_log10_median="median", train_log10_max="max")
        .reset_index()
    )
    test_summary = (
        predictions.groupby("True Composition")["true_log10_conductivity"]
        .agg(test_count="count", test_log10_min="min", test_log10_median="median", test_log10_max="max")
        .reset_index()
    )
    duplicate_summary = train_summary.merge(test_summary, on="True Composition", how="outer").fillna(
        {
            "train_count": 0,
            "test_count": 0,
        }
    )
    duplicate_summary["total_count"] = duplicate_summary["train_count"] + duplicate_summary["test_count"]
    duplicate_summary["train_log10_range"] = duplicate_summary["train_log10_max"] - duplicate_summary["train_log10_min"]
    duplicate_summary["test_log10_range"] = duplicate_summary["test_log10_max"] - duplicate_summary["test_log10_min"]
    duplicate_summary["train_test_median_gap"] = (
        duplicate_summary["test_log10_median"] - duplicate_summary["train_log10_median"]
    ).abs()
    return duplicate_summary.sort_values(
        ["total_count", "train_test_median_gap", "train_log10_range", "test_log10_range"],
        ascending=False,
    )


def analyze_errors(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)

    predictions = pd.read_csv(args.predictions)
    predictions["log10_error"] = predictions["pred_log10_conductivity"] - predictions["true_log10_conductivity"]
    predictions["abs_log10_error"] = predictions["log10_error"].abs()
    predictions["error_factor"] = np.power(10.0, predictions["abs_log10_error"])
    predictions["anion_system"] = predictions["True Composition"].apply(infer_anion_system)

    ordered = predictions.sort_values("abs_log10_error", ascending=False)
    ordered.to_csv(args.output_dir / "test_error_details.csv", index=False)
    ordered.head(args.top_n).to_csv(args.output_dir / "top_test_errors.csv", index=False)

    by_system = (
        predictions.groupby("anion_system")
        .agg(
            rows=("ID", "count"),
            rmse_log10=("log10_error", lambda values: float(np.sqrt(np.mean(np.square(values))))),
            mae_log10=("abs_log10_error", "mean"),
            median_abs_log10_error=("abs_log10_error", "median"),
            max_abs_log10_error=("abs_log10_error", "max"),
        )
        .reset_index()
        .sort_values(["rmse_log10", "rows"], ascending=False)
    )
    by_system.to_csv(args.output_dir / "error_by_anion_system.csv", index=False)

    by_qualifier = (
        predictions.groupby("conductivity_qualifier")
        .agg(
            rows=("ID", "count"),
            rmse_log10=("log10_error", lambda values: float(np.sqrt(np.mean(np.square(values))))),
            mae_log10=("abs_log10_error", "mean"),
            median_abs_log10_error=("abs_log10_error", "median"),
            max_abs_log10_error=("abs_log10_error", "max"),
        )
        .reset_index()
        .sort_values(["rmse_log10", "rows"], ascending=False)
    )
    by_qualifier.to_csv(args.output_dir / "error_by_label_qualifier.csv", index=False)

    duplicate_summary = build_duplicate_summary(args.train, predictions, args)
    duplicate_summary.to_csv(args.output_dir / "duplicate_composition_summary.csv", index=False)

    summary = summarize_errors(predictions)
    with (args.output_dir / "error_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
        file.write("\n")

    print(f"Rows: {summary['rows']}")
    print(f"RMSE(log10): {summary['rmse_log10']:.4f}")
    print(f"MAE(log10): {summary['mae_log10']:.4f}")
    print(f"Abs error > 2 log10 count: {summary['abs_error_gt_2_count']}")
    print(f"Outputs: {args.output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze LightGBM test-set errors.")
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--upper-bound-threshold", type=float, default=1e-10)
    parser.add_argument("--upper-bound-replacement", type=float, default=1e-15)
    return parser.parse_args()


def main() -> None:
    analyze_errors(parse_args())


if __name__ == "__main__":
    main()
