from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib_cache"))

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold


ROOT = Path(__file__).resolve().parent
TARGET_COLUMN = "Ionic conductivity (S cm-1)"
METADATA_COLUMNS = ["ID", "True Composition", "Z_by_element"]
DEFAULT_TRAIN = ROOT / "rawdata" / "feature_train.csv"
DEFAULT_TEST = ROOT / "rawdata" / "feature_test.csv"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "lightgbm"
NUMBER_PATTERN = re.compile(r"^[<>=~≤≥\s]*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)")


def parse_conductivity(value: object) -> tuple[float, str]:
    if pd.isna(value):
        return math.nan, "missing"
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value), "exact"

    text = str(value).strip().replace("−", "-")
    qualifier = "exact"
    if text.startswith(("<", "≤")):
        qualifier = "upper_bound"
    elif text.startswith((">", "≥")):
        qualifier = "lower_bound"

    match = NUMBER_PATTERN.match(text.replace(",", ""))
    if not match:
        return math.nan, "unparsed"
    return float(match.group(1)), qualifier


def add_target_columns(
    data: pd.DataFrame,
    upper_bound_threshold: float,
    upper_bound_replacement: float,
) -> pd.DataFrame:
    values = data[TARGET_COLUMN].apply(parse_conductivity)
    result = data.copy()
    result["conductivity_value"] = [item[0] for item in values]
    result["conductivity_qualifier"] = [item[1] for item in values]
    replace_mask = (
        (result["conductivity_qualifier"] == "upper_bound")
        & (result["conductivity_value"] <= upper_bound_threshold)
    )
    result["conductivity_used"] = result["conductivity_value"].mask(replace_mask, upper_bound_replacement)
    result["log10_conductivity"] = np.log10(result["conductivity_used"].where(result["conductivity_used"] > 0))
    return result


def build_feature_matrix(
    data: pd.DataFrame,
    feature_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    excluded = set(
        METADATA_COLUMNS
        + [
            TARGET_COLUMN,
            "conductivity_value",
            "conductivity_used",
            "conductivity_qualifier",
            "log10_conductivity",
        ]
    )
    drop_all_nan = feature_columns is None
    if feature_columns is None:
        feature_columns = [column for column in data.columns if column not in excluded]

    features = data.reindex(columns=feature_columns).apply(pd.to_numeric, errors="coerce")
    features = features.replace([np.inf, -np.inf], np.nan)
    if drop_all_nan:
        features = features.dropna(axis=1, how="all")
        feature_columns = features.columns.tolist()
    return features, feature_columns


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse_log10": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae_log10": float(mean_absolute_error(y_true, y_pred)),
        "r2_log10": float(r2_score(y_true, y_pred)),
    }


def make_model(random_state: int, n_estimators: int, learning_rate: float) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="regression",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=15,
        min_child_samples=10,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=random_state,
        n_jobs=-1,
        verbosity=-1,
    )


def cross_validate(
    features: pd.DataFrame,
    target: pd.Series,
    ids: pd.Series,
    n_splits: int,
    random_state: int,
    learning_rate: float,
) -> tuple[pd.DataFrame, dict[str, float], list[int]]:
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    predictions: list[pd.DataFrame] = []
    best_iterations: list[int] = []

    for fold, (train_index, valid_index) in enumerate(splitter.split(features), start=1):
        x_train = features.iloc[train_index]
        x_valid = features.iloc[valid_index]
        y_train = target.iloc[train_index]
        y_valid = target.iloc[valid_index]

        model = make_model(random_state=random_state + fold, n_estimators=5000, learning_rate=learning_rate)
        model.fit(
            x_train,
            y_train,
            eval_set=[(x_valid, y_valid)],
            eval_metric="rmse",
            callbacks=[
                lgb.early_stopping(stopping_rounds=100, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        best_iteration = model.best_iteration_ or model.n_estimators
        best_iterations.append(int(best_iteration))
        y_pred = model.predict(x_valid, num_iteration=best_iteration)

        predictions.append(
            pd.DataFrame(
                {
                    "ID": ids.iloc[valid_index].to_numpy(),
                    "fold": fold,
                    "true_log10_conductivity": y_valid.to_numpy(),
                    "pred_log10_conductivity": y_pred,
                    "true_conductivity": np.power(10.0, y_valid.to_numpy()),
                    "pred_conductivity": np.power(10.0, y_pred),
                }
            )
        )

    cv_predictions = pd.concat(predictions, ignore_index=True)
    metrics = regression_metrics(
        cv_predictions["true_log10_conductivity"].to_numpy(),
        cv_predictions["pred_log10_conductivity"].to_numpy(),
    )
    return cv_predictions, metrics, best_iterations


def write_json(path: Path, payload: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")


def train_lightgbm(args: argparse.Namespace) -> None:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    train_raw = pd.read_csv(args.train)
    train_data = add_target_columns(
        train_raw,
        upper_bound_threshold=args.upper_bound_threshold,
        upper_bound_replacement=args.upper_bound_replacement,
    )
    train_data = train_data.dropna(subset=["log10_conductivity"]).reset_index(drop=True)

    x_train, feature_columns = build_feature_matrix(train_data)
    y_train = train_data["log10_conductivity"]

    print(f"Train rows: {len(train_data)}")
    print(f"Feature columns: {len(feature_columns)}")
    replaced_upper_bounds = int(
        (
            (train_data["conductivity_qualifier"] == "upper_bound")
            & (train_data["conductivity_value"] <= args.upper_bound_threshold)
        ).sum()
    )
    print(f"Target upper-bound values replaced with {args.upper_bound_replacement:g}: {replaced_upper_bounds}")

    best_iterations: list[int] = []
    cv_metrics: dict[str, float] | None = None
    cv_predictions_path = output_dir / "cv_predictions.csv"
    if args.run_cv:
        print(f"Running {args.folds}-fold LightGBM CV...")
        cv_predictions, cv_metrics, best_iterations = cross_validate(
            features=x_train,
            target=y_train,
            ids=train_data["ID"],
            n_splits=args.folds,
            random_state=args.random_state,
            learning_rate=args.learning_rate,
        )
        cv_predictions.to_csv(cv_predictions_path, index=False)
        final_estimators = max(50, int(np.median(best_iterations)))
    else:
        print("Skipping random CV; training on the existing train split only.")
        final_estimators = args.n_estimators
        if cv_predictions_path.exists():
            cv_predictions_path.unlink()

    model = make_model(
        random_state=args.random_state,
        n_estimators=final_estimators,
        learning_rate=args.learning_rate,
    )
    model.fit(x_train, y_train)
    joblib.dump(model, output_dir / "lightgbm_model.joblib")

    feature_importance = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance_gain": model.booster_.feature_importance(importance_type="gain"),
            "importance_split": model.booster_.feature_importance(importance_type="split"),
        }
    ).sort_values(["importance_gain", "importance_split"], ascending=False)
    feature_importance.to_csv(output_dir / "feature_importance.csv", index=False)

    metrics: dict[str, object] = {
        "target": "log10(Ionic conductivity (S cm-1))",
        "train_rows": int(len(train_data)),
        "feature_count": int(len(feature_columns)),
        "cv_enabled": bool(args.run_cv),
        "final_n_estimators": int(final_estimators),
        "learning_rate": float(args.learning_rate),
        "target_upper_bound_count": int((train_data["conductivity_qualifier"] == "upper_bound").sum()),
        "upper_bound_threshold": float(args.upper_bound_threshold),
        "upper_bound_replacement": float(args.upper_bound_replacement),
        "upper_bound_replaced_count": replaced_upper_bounds,
    }
    if args.run_cv:
        metrics["cv_folds"] = int(args.folds)
        metrics["cv_metrics"] = cv_metrics
        metrics["cv_best_iterations"] = best_iterations

    if args.test.exists():
        test_raw = pd.read_csv(args.test)
        test_data = add_target_columns(
            test_raw,
            upper_bound_threshold=args.upper_bound_threshold,
            upper_bound_replacement=args.upper_bound_replacement,
        )
        x_test, _ = build_feature_matrix(test_data, feature_columns)
        test_pred_log = model.predict(x_test)
        test_predictions = test_data[[column for column in METADATA_COLUMNS if column in test_data.columns]].copy()
        test_predictions["true_conductivity_raw"] = test_data[TARGET_COLUMN]
        test_predictions["true_conductivity_used"] = test_data["conductivity_used"]
        test_predictions["conductivity_qualifier"] = test_data["conductivity_qualifier"]
        test_predictions["true_log10_conductivity"] = test_data["log10_conductivity"]
        test_predictions["pred_log10_conductivity"] = test_pred_log
        test_predictions["pred_conductivity"] = np.power(10.0, test_pred_log)
        test_predictions.to_csv(output_dir / "test_predictions.csv", index=False)

        valid_test = test_predictions["true_log10_conductivity"].notna()
        if valid_test.any():
            metrics["test_rows"] = int(valid_test.sum())
            metrics["test_metrics"] = regression_metrics(
                test_predictions.loc[valid_test, "true_log10_conductivity"].to_numpy(),
                test_predictions.loc[valid_test, "pred_log10_conductivity"].to_numpy(),
            )

    write_json(output_dir / "metrics.json", metrics)

    print("Done.")
    if cv_metrics is not None:
        print(f"CV RMSE(log10): {cv_metrics['rmse_log10']:.4f}")
        print(f"CV MAE(log10): {cv_metrics['mae_log10']:.4f}")
        print(f"CV R2(log10): {cv_metrics['r2_log10']:.4f}")
    if "test_metrics" in metrics:
        test_metrics = metrics["test_metrics"]
        print(f"Test RMSE(log10): {test_metrics['rmse_log10']:.4f}")
        print(f"Test MAE(log10): {test_metrics['mae_log10']:.4f}")
        print(f"Test R2(log10): {test_metrics['r2_log10']:.4f}")
    print(f"Outputs: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a LightGBM model for ionic conductivity.")
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--run-cv", action="store_true", help="Run random K-fold CV as an optional diagnostic.")
    parser.add_argument("--n-estimators", type=int, default=200, help="Final model tree count when CV is skipped.")
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--upper-bound-threshold", type=float, default=1e-10)
    parser.add_argument("--upper-bound-replacement", type=float, default=1e-15)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    train_lightgbm(parse_args())


if __name__ == "__main__":
    main()
