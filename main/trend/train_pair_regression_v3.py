"""Train conditional trend-v3 regressors on leakage-safe material pairs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

if __package__ is None:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

import joblib
import numpy as np
import optuna
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

from main.paths import RUNS_DIR, portable_path
from main.trend.features import (
    ABSOLUTE_DELTA_BY_DESCRIPTOR,
    SIGNED_DELTA_FEATURES,
)
from main.features import normalize_family
from main.trend.regression_v3 import (
    BASELINE_LOG10_COLUMN,
    DEFAULT_RUN_NAME,
    LOG_RATIO_COLUMN,
    MODEL_FEATURE_COLUMNS_V3,
    SPLIT_GROUP_COLUMN,
    TARGET_COLUMN,
    WEIGHT_COLUMN,
    PairRegressionConfig,
    prepare_regression_data,
    write_json,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)


MODEL_NAMES = ("lightgbm", "catboost", "xgboost", "random_forest")
RUN_NAME = DEFAULT_RUN_NAME
FAMILY_COLUMN = "family"


@dataclass(frozen=True)
class PairRegressorConfig:
    output_root: Path = RUNS_DIR / "trend"
    run_name: str = RUN_NAME
    models: tuple[str, ...] = MODEL_NAMES
    n_trials: int = 50
    catboost_trials: int = 10
    xgboost_trials: int = 20
    random_forest_trials: int = 10
    cv_splits: int = 5
    seed: int = 42
    n_jobs: int = 4


def _reverse(frame: pd.DataFrame) -> pd.DataFrame:
    reverse = frame.copy()
    for descriptor in ABSOLUTE_DELTA_BY_DESCRIPTOR:
        reverse[f"a_{descriptor}"] = frame[f"b_{descriptor}"].to_numpy(float)
        reverse[f"b_{descriptor}"] = frame[f"a_{descriptor}"].to_numpy(float)
    reverse.loc[:, SIGNED_DELTA_FEATURES] = -frame[
        SIGNED_DELTA_FEATURES
    ].to_numpy(float)
    reverse[BASELINE_LOG10_COLUMN] = frame[TARGET_COLUMN].to_numpy(float)
    reverse[TARGET_COLUMN] = frame[BASELINE_LOG10_COLUMN].to_numpy(float)
    reverse[LOG_RATIO_COLUMN] = -frame[LOG_RATIO_COLUMN].to_numpy(float)
    return reverse


def _augment(frame: pd.DataFrame) -> pd.DataFrame:
    forward = frame.copy()
    reverse = _reverse(frame)
    base_weight = frame[WEIGHT_COLUMN].to_numpy(float)
    if not np.isfinite(base_weight).all() or np.any(base_weight <= 0):
        raise ValueError("Pair weights must be finite and positive.")
    # Keep group-equal weighting, but normalize its scale before fitting. A
    # small absolute weight scale can make tree regularization suppress every
    # split and produce a constant predictor.
    base_weight = base_weight / float(np.mean(base_weight))
    forward["training_weight"] = base_weight * 0.5
    reverse["training_weight"] = base_weight * 0.5
    augmented = pd.concat([forward, reverse], ignore_index=True)
    augmented["training_weight"] /= float(augmented["training_weight"].mean())
    return augmented


def _matrices(
    fit: pd.DataFrame,
    other: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, pd.Series, list[str]]:
    """Build numeric descriptors plus one-hot family columns.

    Family categories are fitted on the training fold only.  An unseen
    inference family is represented by an all-zero one-hot vector.

    Returns
    -------
    fit_matrix, other_matrix
        Float32 arrays ready for model.fit / model.predict.
    medians
        Series of per-feature training medians keyed by full feature name
        (numeric features first, then family__ columns with value 0.0).
        Used for NaN imputation at inference time.
    family_categories
        Sorted list of family strings seen in the training fold.  Stored
        explicitly rather than via Series.attrs so it survives all pandas
        operations and downstream pickling without loss.
    """
    fit_values = fit[MODEL_FEATURE_COLUMNS_V3].apply(pd.to_numeric, errors="coerce")
    other_values = other[MODEL_FEATURE_COLUMNS_V3].apply(pd.to_numeric, errors="coerce")
    medians = fit_values.median().fillna(0.0)
    fit_numeric = fit_values.fillna(medians).to_numpy(np.float32)
    other_numeric = other_values.fillna(medians).to_numpy(np.float32)
    fit_family = fit[FAMILY_COLUMN].map(normalize_family).astype(str)
    other_family = other[FAMILY_COLUMN].map(normalize_family).astype(str)
    categories: list[str] = sorted(set(fit_family))
    family_columns = [f"family__{cat}" for cat in categories]
    fit_codes = pd.get_dummies(fit_family).reindex(columns=categories, fill_value=0)
    other_codes = pd.get_dummies(other_family).reindex(columns=categories, fill_value=0)
    fit_matrix = np.concatenate(
        [fit_numeric, fit_codes.to_numpy(np.float32)], axis=1
    )
    other_matrix = np.concatenate(
        [other_numeric, other_codes.to_numpy(np.float32)], axis=1
    )
    matrix_medians = pd.concat([
        medians,
        pd.Series(0.0, index=family_columns),
    ])
    return fit_matrix, other_matrix, matrix_medians, categories


def _parameters(trial: optuna.Trial, model_name: str) -> dict[str, Any]:
    if model_name == "lightgbm":
        depth = trial.suggest_categorical("max_depth", [-1, 3, 4, 5, 6, 8])
        leaves = 63 if depth == -1 else min(63, 2**depth)
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 600, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 7, leaves),
            "max_depth": depth,
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
        }
    if model_name == "catboost":
        return {
            "iterations": trial.suggest_int("iterations", 200, 700, step=100),
            "depth": trial.suggest_int("depth", 4, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 30.0, log=True),
            "random_strength": trial.suggest_float("random_strength", 1e-3, 10.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 5.0),
        }
    if model_name == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 600, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "min_child_weight": trial.suggest_float("min_child_weight", 0.5, 20.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 1e-8, 5.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
        }
    if model_name == "random_forest":
        feature_kind = trial.suggest_categorical(
            "max_features_kind", ["sqrt", "log2", "float"]
        )
        max_features: str | float = feature_kind
        if feature_kind == "float":
            max_features = trial.suggest_float("max_features_float", 0.3, 1.0)
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 700, step=100),
            "max_depth": trial.suggest_categorical(
                "max_depth", [None, 6, 8, 10, 15, 20]
            ),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": max_features,
        }
    raise ValueError(f"Unsupported model: {model_name}")


def _model(
    model_name: str,
    params: dict[str, Any],
    seed: int,
    n_jobs: int,
) -> Any:
    if model_name == "lightgbm":
        from lightgbm import LGBMRegressor

        return LGBMRegressor(
            objective="regression",
            random_state=seed,
            n_jobs=n_jobs,
            verbosity=-1,
            subsample_freq=1,
            **params,
        )
    if model_name == "catboost":
        from catboost import CatBoostRegressor

        return CatBoostRegressor(
            loss_function="RMSE",
            random_seed=seed,
            thread_count=n_jobs,
            verbose=False,
            allow_writing_files=False,
            **params,
        )
    if model_name == "xgboost":
        from xgboost import XGBRegressor

        return XGBRegressor(
            objective="reg:squarederror",
            eval_metric="mae",
            tree_method="hist",
            random_state=seed,
            n_jobs=n_jobs,
            **params,
        )
    if model_name == "random_forest":
        return RandomForestRegressor(random_state=seed, n_jobs=n_jobs, **params)
    raise ValueError(f"Unsupported model: {model_name}")


def _fit_predict(
    model_name: str,
    params: dict[str, Any],
    fit: pd.DataFrame,
    other: pd.DataFrame,
    config: PairRegressorConfig,
    seed: int,
) -> tuple[Any, pd.Series, list[str], np.ndarray]:
    augmented = _augment(fit)
    fit_x, other_x, medians, family_categories = _matrices(augmented, other)
    estimator = _model(model_name, params, seed, config.n_jobs)
    estimator.fit(
        fit_x,
        augmented[TARGET_COLUMN].to_numpy(float),
        sample_weight=augmented["training_weight"].to_numpy(float),
    )
    prediction = np.asarray(estimator.predict(other_x), dtype=float)
    return estimator, medians, family_categories, prediction


def _metrics(frame: pd.DataFrame, prediction: np.ndarray) -> dict[str, float]:
    target_log_sigma_b = frame[TARGET_COLUMN].to_numpy(float)
    baseline_log_sigma_a = frame[BASELINE_LOG10_COLUMN].to_numpy(float)
    true_log_ratio = target_log_sigma_b - baseline_log_sigma_a
    predicted_log_ratio = prediction - baseline_log_sigma_a
    weights = frame[WEIGHT_COLUMN].to_numpy(float)
    residual = prediction - target_log_sigma_b
    true_sigma_b = np.power(10.0, target_log_sigma_b)
    predicted_sigma_b = np.power(10.0, prediction)
    absolute_change_residual = predicted_sigma_b - true_sigma_b
    correlation = (
        spearmanr(true_log_ratio, predicted_log_ratio).statistic
        if np.std(true_log_ratio) > 0 and np.std(predicted_log_ratio) > 0
        else math.nan
    )
    group_mae = [
        float(np.mean(np.abs(residual[np.asarray(list(indices), dtype=int)])))
        for indices in frame.groupby("group_id").groups.values()
    ]
    true_direction = np.sign(
        np.where(np.abs(true_log_ratio) < 0.1, 0.0, true_log_ratio)
    )
    predicted_direction = np.sign(
        np.where(np.abs(predicted_log_ratio) < 0.1, 0.0, predicted_log_ratio)
    )
    increase_true = true_log_ratio > 0.1
    increase_predicted = predicted_log_ratio > 0.1
    # Top-1 is a recommendation metric: rank pairs by predicted change ratio,
    # then check whether the selected pair has the largest observed change.
    # Ranking by predicted log(sigma_B) would answer a different question.
    predicted_delta = prediction - baseline_log_sigma_a
    group_top1_hits = []
    for _, group in frame.assign(_predicted_delta=predicted_delta).groupby("group_id"):
        predicted_best = group.iloc[
            int(np.argmax(group["_predicted_delta"].to_numpy(float)))
        ][LOG_RATIO_COLUMN]
        actual_best = group[LOG_RATIO_COLUMN].max()
        group_top1_hits.append(float(predicted_best == actual_best))
    return {
        "weighted_mae_delta_log10_ratio": float(
            mean_absolute_error(true_log_ratio, predicted_log_ratio, sample_weight=weights)
        ),
        "weighted_rmse_delta_log10_ratio": float(
            math.sqrt(
                mean_squared_error(
                    true_log_ratio, predicted_log_ratio, sample_weight=weights
                )
            )
        ),
        "mae_delta_log10_ratio": float(
            mean_absolute_error(true_log_ratio, predicted_log_ratio)
        ),
        "weighted_mae_delta_conductivity_S_cm-1": float(
            np.average(np.abs(absolute_change_residual), weights=weights)
        ),
        "mae_delta_conductivity_S_cm-1": float(
            np.mean(np.abs(absolute_change_residual))
        ),
        "r2_log10_sigma_b": float(
            r2_score(target_log_sigma_b, prediction, sample_weight=weights)
        ),
        "spearman_delta_log10_ratio": float(correlation),
        "group_macro_mae_delta_log10_ratio": float(np.mean(group_mae)),
        "direction_accuracy_at_0.1_log10": float(
            np.mean(true_direction == predicted_direction)
        ),
        "median_absolute_error_delta_log10_ratio": float(
            np.median(np.abs(residual))
        ),
        "p90_absolute_error_delta_log10_ratio": float(
            np.quantile(np.abs(residual), 0.9)
        ),
        "within_two_fold_ratio": float(
            np.mean(np.abs(true_log_ratio - predicted_log_ratio) <= math.log10(2.0))
        ),
        "increase_precision_at_0.1_log10": float(
            np.sum(increase_true & increase_predicted)
            / max(np.sum(increase_predicted), 1)
        ),
        "increase_recall_at_0.1_log10": float(
            np.sum(increase_true & increase_predicted)
            / max(np.sum(increase_true), 1)
        ),
        "group_top1_accuracy": float(np.mean(group_top1_hits))
        if group_top1_hits
        else math.nan,
    }


def _folds(frame: pd.DataFrame, config: PairRegressorConfig) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = GroupKFold(
        n_splits=config.cv_splits, shuffle=True, random_state=config.seed
    )
    return list(
        splitter.split(frame, groups=frame[SPLIT_GROUP_COLUMN].astype(str))
    )


def _cross_validate(
    model_name: str,
    params: dict[str, Any],
    train: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    config: PairRegressorConfig,
    trial: optuna.Trial | None = None,
) -> tuple[float, list[dict[str, float]]]:
    fold_metrics = []
    for fold_number, (fit_index, valid_index) in enumerate(folds, 1):
        fit = train.iloc[fit_index].reset_index(drop=True)
        valid = train.iloc[valid_index].reset_index(drop=True)
        _, _, _, prediction = _fit_predict(
            model_name, params, fit, valid, config, config.seed + fold_number
        )
        metrics = _metrics(valid, prediction)
        metrics["fold"] = float(fold_number)
        fold_metrics.append(metrics)
        running = float(
            np.mean([item["weighted_mae_delta_log10_ratio"] for item in fold_metrics])
        )
        if trial is not None:
            trial.report(running, fold_number)
            if trial.should_prune():
                raise optuna.TrialPruned()
    score = float(
        np.mean([item["weighted_mae_delta_log10_ratio"] for item in fold_metrics])
    )
    return score, fold_metrics


def _tune(
    model_name: str,
    train: pd.DataFrame,
    config: PairRegressorConfig,
    model_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], int]:
    folds = _folds(train, config)
    study = optuna.create_study(
        study_name=f"formula_pair_{model_name}_{config.seed}",
        storage=f"sqlite:///{(model_dir / 'optuna.db').resolve()}",
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=config.seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2),
        load_if_exists=True,
    )
    target_trials = {
        "catboost": config.catboost_trials,
        "xgboost": config.xgboost_trials,
        "random_forest": config.random_forest_trials,
    }.get(model_name, config.n_trials)
    trials_before = len(study.trials)
    remaining = max(target_trials - trials_before, 0)
    start = time.monotonic()

    def objective(trial: optuna.Trial) -> float:
        score, _ = _cross_validate(
            model_name, _parameters(trial, model_name), train, folds, config, trial
        )
        return score

    def progress(study_object: optuna.Study, trial: optuna.FrozenTrial) -> None:
        finished = max(len(study_object.trials) - trials_before, 1)
        elapsed = time.monotonic() - start
        eta = max(remaining - finished, 0) * elapsed / finished
        print(
            f"[{model_name}] {finished}/{remaining}; "
            f"best_cv_mae={study_object.best_value:.4f}; "
            f"elapsed={elapsed / 60:.1f}min; eta={eta / 60:.1f}min",
            flush=True,
        )

    if remaining:
        study.optimize(
            objective,
            n_trials=remaining,
            callbacks=[progress],
            show_progress_bar=False,
        )
    params = _parameters(optuna.trial.FixedTrial(study.best_trial.params), model_name)
    score, fold_metrics = _cross_validate(
        model_name, params, train, folds, config
    )
    cv_metrics = {
        "weighted_mae_delta_log10_ratio": score,
        "folds": fold_metrics,
    }
    write_json(model_dir / "best_params.json", params)
    write_json(model_dir / "cv_metrics.json", cv_metrics)
    study.trials_dataframe().to_csv(model_dir / "optuna_trials.csv", index=False)
    return params, cv_metrics, int(study.best_trial.number)


def _prediction_table(
    frame: pd.DataFrame,
    prediction: np.ndarray,
) -> pd.DataFrame:
    result = frame[[
        "pair_id",
        "group_id",
        SPLIT_GROUP_COLUMN,
        "doi",
        "family",
        "formula_a",
        "formula_b",
        BASELINE_LOG10_COLUMN,
        TARGET_COLUMN,
        LOG_RATIO_COLUMN,
        WEIGHT_COLUMN,
    ]].copy()
    sigma_a = np.power(10.0, frame[BASELINE_LOG10_COLUMN].to_numpy(float))
    predicted_sigma_b = np.power(10.0, prediction)
    result["predicted_log10_conductivity_b"] = prediction
    result["predicted_conductivity_b_S_cm-1"] = predicted_sigma_b
    result["predicted_delta_conductivity_S_cm-1"] = predicted_sigma_b - sigma_a
    result["predicted_absolute_delta_conductivity_S_cm-1"] = np.abs(
        predicted_sigma_b - sigma_a
    )
    result["predicted_delta_log10_ratio"] = (
        prediction - frame[BASELINE_LOG10_COLUMN].to_numpy(float)
    )
    result["predicted_ratio_b_over_a"] = np.power(
        10.0, result["predicted_delta_log10_ratio"].to_numpy(float)
    )
    result["absolute_error_delta_log10_ratio"] = np.abs(
        prediction - frame[TARGET_COLUMN].to_numpy(float)
    )
    return result


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = (np.cumsum(weights) - 0.5 * weights) / weights.sum()
    return float(np.interp(quantile, cumulative, values))


def train_pair_regression_v3(
    pair_config: PairRegressionConfig | None = None,
    model_config: PairRegressorConfig | None = None,
) -> dict[str, Any]:
    """Run data preparation, tuning, validation selection, and final testing."""

    pair_config = pair_config or PairRegressionConfig()
    model_config = model_config or PairRegressorConfig()
    unknown = sorted(set(model_config.models) - set(MODEL_NAMES))
    if unknown:
        raise ValueError(f"Unsupported models: {unknown}")
    run_dir = model_config.output_root / model_config.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    data_dir = run_dir / "data"
    cached_paths = {
        name: data_dir / f"{name}.csv"
        for name in ("train", "validation", "test")
    }
    manifest_path = data_dir / "data_manifest.json"
    if manifest_path.exists() and all(path.exists() for path in cached_paths.values()):
        splits = {
            name: pd.read_csv(path) for name, path in cached_paths.items()
        }
        data_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        cached_features = data_manifest.get("model_features")
        cached_target = data_manifest.get("pairing", {}).get("target", {}).get("name")
        if cached_features != MODEL_FEATURE_COLUMNS_V3 or cached_target != TARGET_COLUMN:
            raise ValueError(
                "Cached pair data does not match the conditional target or F43 "
                "feature schema; use a new run_name."
            )
        print("Reusing prepared pair features", flush=True)
    else:
        splits, data_manifest = prepare_regression_data(pair_config, data_dir)
    train = splits["train"]
    validation = splits["validation"]
    test = splits["test"]
    candidate_results = []
    best_params_by_model = {}
    validation_tables = {}
    for model_name in model_config.models:
        model_dir = run_dir / model_name
        model_dir.mkdir(exist_ok=True)
        print(f"Training {model_name}", flush=True)
        start = time.monotonic()
        params, cv_metrics, best_trial = _tune(
            model_name, train, model_config, model_dir
        )
        estimator, medians, family_categories, prediction = _fit_predict(
            model_name,
            params,
            train,
            validation,
            model_config,
            model_config.seed,
        )
        metrics = _metrics(validation, prediction)
        table = _prediction_table(validation, prediction)
        table.to_csv(model_dir / "validation_predictions.csv", index=False)
        write_json(model_dir / "validation_metrics.json", metrics)
        input_features = list(medians.index)
        joblib.dump({
            "model": estimator,
            "model_name": model_name,
            "numeric_features": MODEL_FEATURE_COLUMNS_V3,
            "input_features": input_features,
            "family_categories": family_categories,
            "numeric_medians": medians.to_dict(),
            "reverse_training_augmentation": True,
        }, model_dir / "model.joblib")
        if hasattr(estimator, "feature_importances_"):
            pd.DataFrame({
                "feature": input_features,
                "importance": np.asarray(estimator.feature_importances_, dtype=float),
            }).sort_values("importance", ascending=False).to_csv(
                model_dir / "feature_importance.csv", index=False
            )
        candidate_results.append({
            "model": model_name,
            "best_trial": best_trial,
            "cv_weighted_mae_delta_log10_ratio": cv_metrics[
                "weighted_mae_delta_log10_ratio"
            ],
            **{f"validation_{key}": value for key, value in metrics.items()},
            "elapsed_minutes": (time.monotonic() - start) / 60.0,
        })
        best_params_by_model[model_name] = params
        validation_tables[model_name] = table

    comparison = pd.DataFrame(candidate_results).sort_values(
        "validation_weighted_mae_delta_log10_ratio", ignore_index=True
    )
    comparison.to_csv(run_dir / "model_comparison.csv", index=False)
    best_name = str(comparison.iloc[0]["model"])
    validation_zero_metrics = _metrics(
        validation, validation[BASELINE_LOG10_COLUMN].to_numpy(float)
    )
    calibration = validation_tables[best_name]
    interval = _weighted_quantile(
        calibration["absolute_error_delta_log10_ratio"].to_numpy(float),
        calibration[WEIGHT_COLUMN].to_numpy(float),
        0.90,
    )
    train_validation = pd.concat([train, validation], ignore_index=True)
    estimator, medians, family_categories, prediction = _fit_predict(
        best_name,
        best_params_by_model[best_name],
        train_validation,
        test,
        model_config,
        model_config.seed,
    )
    test_metrics = _metrics(test, prediction)
    test_zero_metrics = _metrics(
        test, test[BASELINE_LOG10_COLUMN].to_numpy(float)
    )
    test_metrics[
        "zero_change_baseline_weighted_mae_delta_log10_ratio"
    ] = test_zero_metrics[
        "weighted_mae_delta_log10_ratio"
    ]
    test_metrics["beats_zero_change_baseline_weighted_mae"] = (
        test_metrics["weighted_mae_delta_log10_ratio"]
        < test_zero_metrics["weighted_mae_delta_log10_ratio"]
    )
    test_metrics[
        "validation_calibrated_90pct_error_delta_log10_ratio"
    ] = interval
    best_dir = run_dir / "best_model"
    best_dir.mkdir(exist_ok=True)
    test_table = _prediction_table(test, prediction)
    lower_log_sigma_b = prediction - interval
    upper_log_sigma_b = prediction + interval
    baseline_log_sigma_a = test[BASELINE_LOG10_COLUMN].to_numpy(float)
    sigma_a = np.power(10.0, baseline_log_sigma_a)
    test_table["interval_log10_conductivity_b_lower_90"] = lower_log_sigma_b
    test_table["interval_log10_conductivity_b_upper_90"] = upper_log_sigma_b
    test_table["interval_delta_log10_ratio_lower_90"] = (
        lower_log_sigma_b - baseline_log_sigma_a
    )
    test_table["interval_delta_log10_ratio_upper_90"] = (
        upper_log_sigma_b - baseline_log_sigma_a
    )
    test_table["interval_delta_conductivity_lower_90_S_cm-1"] = (
        np.power(10.0, lower_log_sigma_b) - sigma_a
    )
    test_table["interval_delta_conductivity_upper_90_S_cm-1"] = (
        np.power(10.0, upper_log_sigma_b) - sigma_a
    )
    test_table.to_csv(best_dir / "test_predictions.csv", index=False)
    write_json(best_dir / "test_metrics.json", test_metrics)
    feature_min = train_validation[MODEL_FEATURE_COLUMNS_V3].min().to_dict()
    feature_max = train_validation[MODEL_FEATURE_COLUMNS_V3].max().to_dict()
    model_status = (
        "validated_candidate"
        if test_metrics["beats_zero_change_baseline_weighted_mae"]
        and test_metrics["r2_log10_sigma_b"] > 0
        else "experimental_not_for_deployment"
    )
    joblib.dump({
        "model": estimator,
        "model_name": best_name,
        "numeric_features": MODEL_FEATURE_COLUMNS_V3,
        "input_features": list(medians.index),
        "family_categories": family_categories,
        "numeric_medians": medians.to_dict(),
        "feature_min": feature_min,
        "feature_max": feature_max,
        "target": TARGET_COLUMN,
        "target_definition": "log10(sigma_B)",
        "baseline_feature": BASELINE_LOG10_COLUMN,
        "derived_outputs": {
            "delta_conductivity_S_cm-1": "sigma_B - sigma_A",
            "delta_log10_ratio": "log10(sigma_B) - log10(sigma_A)",
        },
        "reverse_training_augmentation": True,
        "weight_normalization": "training weights normalized to mean 1 after A/B augmentation",
        "family_feature": "one-hot family fitted on each training fold",
        "prediction_interval_absolute_error_delta_log10_ratio_90": interval,
        "training_scope": "same-study related formula pairs with measured sigma_A",
        "model_status": model_status,
    }, best_dir / "model.joblib")
    write_json(best_dir / "best_params.json", best_params_by_model[best_name])
    manifest = {
        "run_name": model_config.run_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "pair_config": asdict(pair_config),
        "model_config": asdict(model_config),
        "data": data_manifest,
        "selected_model": best_name,
        "selection_metric": "validation_weighted_mae_delta_log10_ratio",
        "test_metrics": test_metrics,
        "validation_zero_baseline": validation_zero_metrics,
        "test_zero_baseline": test_zero_metrics,
        "model_status": model_status,
        "artifact": portable_path(best_dir / "model.joblib"),
    }
    write_json(run_dir / "manifest.json", manifest)
    columns = [
        "model",
        "cv_weighted_mae_delta_log10_ratio",
        "validation_weighted_mae_delta_log10_ratio",
        "validation_weighted_rmse_delta_log10_ratio",
        "validation_weighted_mae_delta_conductivity_S_cm-1",
        "validation_r2_log10_sigma_b",
        "validation_spearman_delta_log10_ratio",
        "validation_direction_accuracy_at_0.1_log10",
    ]
    report = [
        "# Conditional formula-pair trend-v3 regression report",
        "",
        f"Selected model: **{best_name}**",
        "",
        "The model consumes two formulas plus measured `sigma_A` and predicts `log10(sigma_B)`.",
        "Absolute change and log-ratio change are derived from the positive sigma_B prediction.",
        "Training pairs are same-study comparisons; held-out formulas and DOIs do not leak.",
        "",
        comparison[columns].to_markdown(index=False),
        "",
        "## Held-out test",
        "",
        *[f"- `{key}`: {value}" for key, value in test_metrics.items()],
        "",
        f"Model status: **{model_status}**",
        "",
    ]
    (run_dir / "model_report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Selected model: {best_name}", flush=True)
    print(
        "Test weighted log-ratio MAE: "
        f"{test_metrics['weighted_mae_delta_log10_ratio']:.4f}",
        flush=True,
    )
    print(f"Output: {run_dir.resolve()}", flush=True)
    return {
        "run_dir": run_dir,
        "selected_model": best_name,
        "comparison": comparison,
        "test_metrics": test_metrics,
    }


def main() -> None:
    train_pair_regression_v3()


if __name__ == "__main__":
    main()
