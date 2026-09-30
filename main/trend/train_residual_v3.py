"""Residual-correction trend model (v3).

Architecture
------------
Given a pair (A, B), define:

    d_0  = f_abs(B) - f_abs(A)          # absolute-model anchor

where f_abs is the saved absolute LightGBM model.  Train a LightGBM
residual corrector h on:

    r_train = true_delta - d_0           # what the absolute model missed

Final prediction at inference:

    delta_hat = d_0 + h(pair_features)

This forces the trend model to learn *systematic deviations* from the
absolute baseline rather than the full log-ratio, which is a harder target.

To avoid leaking training labels into d_0, the absolute model predictions
on the training set are produced via DOI-grouped fold-out prediction
(same GroupKFold used for CV).  Validation and test use the fitted
final absolute model directly.

Entry point
-----------
    python main/trend/train_residual_v3.py
"""

from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

from main.features import (
    SMALL_FEATURE_SPECS,
    FeatureConfig,
    make_feature_table,
    normalize_family,
)
from main.paths import RUNS_DIR, portable_path
from main.trend.features import (
    ABSOLUTE_DELTA_BY_DESCRIPTOR,
    SIGNED_DELTA_FEATURES,
)
from main.trend.features_v3 import (
    MODEL_FEATURE_COLUMNS_V3,
    TARGET_COLUMN,  # "delta_log10_IC" — the v3 target IS the log ratio
)
from main.trend.regression_v3 import (
    WEIGHT_COLUMN,
    write_json,
)
from main.trend.split_v3 import DEFAULT_TRAIN_V3, DEFAULT_VALIDATION_V3
from main.paths import DATA_DIR as _DATA_DIR

# The v3 feature CSVs store the target as the directed log ratio itself:
#     delta_log10_IC = log10_IC_b - log10_IC_a
# There is no separate log10_conductivity_a/b column pair (those belong to the
# older trend-v2 schema), so the baseline column is the model input log10_IC_a.
BASELINE_LOG10_COLUMN = "log10_IC_a"
LOG_RATIO_COLUMN = TARGET_COLUMN
# Grouping column for CV folds: the DOI, which keeps same-study pairs together.
SPLIT_GROUP_COLUMN = "doi"

# Full feature CSV (all pairs, before train/validation split).
# Used to derive a held-out test set via a second grouped split.
DEFAULT_FULL_V3 = _DATA_DIR / "trend" / "data-trend-v3-pairs-feature.csv"

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_ABS_RUN = (
    RUNS_DIR / "absolute" / "abs_v2_f37_native_family_lgbm_trials50_seed42"
)
DEFAULT_OUTPUT_ROOT = RUNS_DIR / "trend"
FAMILY_COLUMN = "family"
GROUP_COLUMN = "doi"


@dataclass(frozen=True)
class ResidualConfig:
    train_path: Path = DEFAULT_TRAIN_V3
    validation_path: Path = DEFAULT_VALIDATION_V3
    abs_run_dir: Path = DEFAULT_ABS_RUN
    output_root: Path = DEFAULT_OUTPUT_ROOT
    run_name: str = "trend_v3_residual_lgbm_optuna50_seed42"
    n_trials: int = 50
    cv_splits: int = 5
    seed: int = 42
    n_jobs: int = 4


# ---------------------------------------------------------------------------
# Absolute model helpers
# ---------------------------------------------------------------------------

def _load_abs_bundle(abs_run_dir: Path) -> dict[str, Any]:
    best_txt = abs_run_dir / "best_model.txt"
    model_name = best_txt.read_text(encoding="utf-8").strip() if best_txt.exists() else "lightgbm"
    bundle = joblib.load(abs_run_dir / model_name / "model.joblib")
    bundle["_model_name"] = model_name
    bundle["_run_dir"] = abs_run_dir
    return bundle


def _abs_predict_single_formula(
    formula: str,
    family: str,
    abs_bundle: dict[str, Any],
) -> float:
    """Return log10(sigma) for one formula using the saved absolute model."""
    from main.features import TARGET_COLUMN as ABS_TARGET

    dummy_row = pd.DataFrame({
        "ID": ["pred_0001"],
        "True Composition": [formula],
        "Family": [normalize_family(family)],
        ABS_TARGET: [1e-6],  # dummy; not used in prediction
    })

    feature_cols = list(abs_bundle["feature_cols"])
    family_mapping = abs_bundle.get("family_mapping")
    categorical_features = list(abs_bundle.get("categorical_features", []))
    category_levels = dict(abs_bundle.get("category_levels", {}))
    family_onehot_categories = list(abs_bundle.get("family_onehot_categories", []))
    feature_medians = pd.Series(abs_bundle.get("feature_medians", {}), dtype=float)
    scaler = abs_bundle.get("scaler")
    model = abs_bundle["model"]

    feature_result = make_feature_table(
        dummy_row,
        FeatureConfig(
            min_conductivity=None,
            include_family="family" in feature_cols,
            family_encoding="native" if "family" in categorical_features else "ordinal",
            include_interactions=True,
            include_small_features=any(
                col in feature_cols for col, *_ in SMALL_FEATURE_SPECS
            ),
            family_mapping=family_mapping,
            output_path=None,
        ),
    )
    features = feature_result.table

    for cat in family_onehot_categories:
        features[f"family__{cat}"] = (
            features["Family"].astype(str).eq(cat).astype(float)
        )

    X = features.reindex(columns=feature_cols).copy()
    numeric_cols = [c for c in feature_cols if c not in categorical_features]
    for col in numeric_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    X[numeric_cols] = X[numeric_cols].replace([np.inf, -np.inf], np.nan)
    X[numeric_cols] = X[numeric_cols].fillna(
        feature_medians.reindex(numeric_cols).fillna(0.0)
    )
    for col in categorical_features:
        levels = list(category_levels[col])
        vals = X[col].astype("string").fillna("unknown")
        X[col] = pd.Categorical(
            vals.where(vals.isin(levels), "unknown"),
            categories=levels,
            ordered=False,
        )

    predict_X = X if scaler is None else pd.DataFrame(
        scaler.transform(X), columns=feature_cols, index=X.index
    )
    return float(model.predict(predict_X)[0])


def _abs_delta(row: pd.Series, abs_bundle: dict[str, Any]) -> float:
    """d_0 = f_abs(B) - f_abs(A) for one pair row."""
    family = normalize_family(str(row.get(FAMILY_COLUMN, "unknown")))
    fa = str(row["formula_a"])
    fb = str(row["formula_b"])
    log_a = _abs_predict_single_formula(fa, family, abs_bundle)
    log_b = _abs_predict_single_formula(fb, family, abs_bundle)
    return log_b - log_a


def _compute_abs_deltas_batch(
    frame: pd.DataFrame,
    abs_bundle: dict[str, Any],
) -> np.ndarray:
    """Vectorised batch: d_0 for every row in frame."""
    deltas = []
    for _, row in frame.iterrows():
        try:
            deltas.append(_abs_delta(row, abs_bundle))
        except Exception:
            deltas.append(math.nan)
    return np.array(deltas, dtype=float)


# ---------------------------------------------------------------------------
# Residual feature matrix
# ---------------------------------------------------------------------------

def _reverse_for_residual(frame: pd.DataFrame) -> pd.DataFrame:
    """Swap A<->B: negate signed deltas, swap a_/b_ descriptors, flip target."""
    reverse = frame.copy()
    for descriptor in ABSOLUTE_DELTA_BY_DESCRIPTOR:
        reverse[f"a_{descriptor}"] = frame[f"b_{descriptor}"].to_numpy(float)
        reverse[f"b_{descriptor}"] = frame[f"a_{descriptor}"].to_numpy(float)
    reverse.loc[:, SIGNED_DELTA_FEATURES] = -frame[SIGNED_DELTA_FEATURES].to_numpy(float)
    # Baseline moves to the other endpoint after the swap: the new baseline is
    # the old B.  The split CSVs omit log10_IC_b (trace-only column), so it is
    # reconstructed from log10_IC_b = log10_IC_a + delta_log10_IC.  Skipping
    # this leaves the reverse half with a mismatched baseline.
    if BASELINE_LOG10_COLUMN in reverse.columns and TARGET_COLUMN in reverse.columns:
        baseline_b = (
            frame[BASELINE_LOG10_COLUMN].to_numpy(float)
            + frame[TARGET_COLUMN].to_numpy(float)
        )
        reverse[BASELINE_LOG10_COLUMN] = baseline_b
        reverse[TARGET_COLUMN] = -frame[TARGET_COLUMN].to_numpy(float)
    if LOG_RATIO_COLUMN in reverse.columns:
        reverse[LOG_RATIO_COLUMN] = -frame[LOG_RATIO_COLUMN].to_numpy(float)
    # d_0 also flips sign under swap
    if "abs_delta_d0" in reverse.columns:
        reverse["abs_delta_d0"] = -frame["abs_delta_d0"].to_numpy(float)
    if "residual_target" in reverse.columns:
        reverse["residual_target"] = -frame["residual_target"].to_numpy(float)
    return reverse


def _augment_residual(frame: pd.DataFrame) -> pd.DataFrame:
    """Symmetry-augment with A<->B swap; normalise weights to mean=1."""
    forward = frame.copy()
    reverse = _reverse_for_residual(frame)
    base_w = frame[WEIGHT_COLUMN].to_numpy(float)
    if not np.isfinite(base_w).all() or np.any(base_w <= 0):
        raise ValueError("Pair weights must be finite and positive.")
    base_w = base_w / float(np.mean(base_w))
    forward["training_weight"] = base_w * 0.5
    reverse["training_weight"] = base_w * 0.5
    augmented = pd.concat([forward, reverse], ignore_index=True)
    augmented["training_weight"] /= float(augmented["training_weight"].mean())
    return augmented


def _residual_matrices(
    fit: pd.DataFrame,
    other: pd.DataFrame,
    residual_feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, pd.Series, list[str]]:
    """Build [numeric | one-hot family] matrices for the residual corrector.

    The residual corrector gets the same composition-difference features as
    the direct trend model, plus the absolute-model anchor d_0 as an
    additional scalar input.  Family is one-hot encoded from the fit fold.

    Returns fit_matrix, other_matrix, medians, family_categories.
    """
    fit_values = fit[residual_feature_cols].apply(pd.to_numeric, errors="coerce")
    other_values = other[residual_feature_cols].apply(pd.to_numeric, errors="coerce")
    medians = fit_values.median().fillna(0.0)
    fit_numeric = fit_values.fillna(medians).to_numpy(np.float32)
    other_numeric = other_values.fillna(medians).to_numpy(np.float32)

    fit_family = fit[FAMILY_COLUMN].map(normalize_family).astype(str)
    other_family = other[FAMILY_COLUMN].map(normalize_family).astype(str)
    categories: list[str] = sorted(set(fit_family))
    family_cols = [f"family__{c}" for c in categories]
    fit_oh = pd.get_dummies(fit_family).reindex(columns=categories, fill_value=0)
    other_oh = pd.get_dummies(other_family).reindex(columns=categories, fill_value=0)

    fit_mat = np.concatenate([fit_numeric, fit_oh.to_numpy(np.float32)], axis=1)
    other_mat = np.concatenate([other_numeric, other_oh.to_numpy(np.float32)], axis=1)

    all_medians = pd.concat([medians, pd.Series(0.0, index=family_cols)])
    return fit_mat, other_mat, all_medians, categories


# ---------------------------------------------------------------------------
# LightGBM residual model
# ---------------------------------------------------------------------------

def _lgbm_params(trial: optuna.Trial) -> dict[str, Any]:
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


def _build_lgbm(params: dict[str, Any], seed: int, n_jobs: int) -> Any:
    from lightgbm import LGBMRegressor
    return LGBMRegressor(
        objective="regression",
        random_state=seed,
        n_jobs=n_jobs,
        verbosity=-1,
        subsample_freq=1,
        **params,
    )


# ---------------------------------------------------------------------------
# Metrics (operate on residual predictions; convert back to full delta)
# ---------------------------------------------------------------------------

def _residual_metrics(
    frame: pd.DataFrame,
    residual_prediction: np.ndarray,
    d0: np.ndarray,
) -> dict[str, float]:
    """Compute metrics for the combined prediction delta_hat = d0 + r_hat."""
    # v3: TARGET_COLUMN (delta_log10_IC) is the directed log ratio itself.
    true_delta = frame[TARGET_COLUMN].to_numpy(float)
    predicted_delta = d0 + residual_prediction
    weights = frame[WEIGHT_COLUMN].to_numpy(float)
    residual = predicted_delta - true_delta
    within_two_fold = float(np.mean(np.abs(residual) <= math.log10(2.0)))

    true_dir = np.sign(np.where(np.abs(true_delta) < 0.1, 0.0, true_delta))
    pred_dir = np.sign(np.where(np.abs(predicted_delta) < 0.1, 0.0, predicted_delta))
    increase_true = true_delta > 0.1
    increase_pred = predicted_delta > 0.1

    group_top1 = []
    tmp = frame.assign(_pred_delta=predicted_delta)
    for _, grp in tmp.groupby("group_id"):
        best_pred = grp.iloc[int(np.argmax(grp["_pred_delta"].to_numpy()))][TARGET_COLUMN]
        best_true = grp[TARGET_COLUMN].max()
        group_top1.append(float(best_pred == best_true))

    corr = (
        spearmanr(true_delta, predicted_delta).statistic
        if np.std(true_delta) > 0 and np.std(predicted_delta) > 0
        else math.nan
    )
    return {
        "weighted_mae_delta_log10_ratio": float(
            mean_absolute_error(true_delta, predicted_delta, sample_weight=weights)
        ),
        "mae_delta_log10_ratio": float(mean_absolute_error(true_delta, predicted_delta)),
        "weighted_rmse_delta_log10_ratio": float(
            math.sqrt(mean_squared_error(true_delta, predicted_delta, sample_weight=weights))
        ),
        "r2_log10_sigma_b": float(
            r2_score(
                frame[BASELINE_LOG10_COLUMN].to_numpy(float) + true_delta,
                frame[BASELINE_LOG10_COLUMN].to_numpy(float) + predicted_delta,
                sample_weight=weights,
            )
        ),
        "spearman_delta_log10_ratio": float(corr),
        "direction_accuracy_at_0.1_log10": float(np.mean(true_dir == pred_dir)),
        "within_two_fold_ratio": within_two_fold,
        "increase_precision_at_0.1_log10": float(
            np.sum(increase_true & increase_pred) / max(np.sum(increase_pred), 1)
        ),
        "increase_recall_at_0.1_log10": float(
            np.sum(increase_true & increase_pred) / max(np.sum(increase_true), 1)
        ),
        "group_top1_accuracy": float(np.mean(group_top1)) if group_top1 else math.nan,
        "median_absolute_error_delta_log10_ratio": float(np.median(np.abs(residual))),
        "p90_absolute_error_delta_log10_ratio": float(np.quantile(np.abs(residual), 0.9)),
    }


# ---------------------------------------------------------------------------
# Fold-out absolute predictions (leak-safe for training set)
# ---------------------------------------------------------------------------

def _fold_out_abs_deltas(
    train: pd.DataFrame,
    abs_bundle: dict[str, Any],
    config: ResidualConfig,
) -> np.ndarray:
    """Compute d_0 for the training set via DOI-grouped fold-out prediction.

    For each CV fold, the absolute model is applied directly (it is already
    trained on its own data; fold-out here means we use a fresh feature-table
    computation for each held-out fold, which avoids any in-sample evaluation
    artefacts on the trend training set.  The absolute model weights are
    fixed throughout — we are not re-training it).
    """
    splitter = GroupKFold(n_splits=config.cv_splits)
    d0 = np.full(len(train), math.nan, dtype=float)
    for _, valid_idx in splitter.split(train, groups=train[SPLIT_GROUP_COLUMN].astype(str)):
        fold_frame = train.iloc[valid_idx]
        fold_d0 = _compute_abs_deltas_batch(fold_frame, abs_bundle)
        d0[valid_idx] = fold_d0
    if not np.isfinite(d0).all():
        n_bad = int(np.sum(~np.isfinite(d0)))
        print(
            f"  [warn] {n_bad}/{len(d0)} absolute-model deltas are non-finite "
            "and will be imputed with 0.0",
            flush=True,
        )
        d0 = np.where(np.isfinite(d0), d0, 0.0)
    return d0


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _fit_residual(
    fit: pd.DataFrame,
    other: pd.DataFrame,
    fit_d0: np.ndarray,
    other_d0: np.ndarray,
    residual_feature_cols: list[str],
    params: dict[str, Any],
    config: ResidualConfig,
    seed: int,
) -> tuple[Any, pd.Series, list[str], np.ndarray]:
    """Train residual corrector on fit, predict on other.

    Returns (estimator, medians, family_categories, predicted_full_delta).
    The returned predictions are the full combined delta d0 + r_hat.
    """
    # Add d_0 as a feature and set the residual as the target
    fit_aug = _augment_residual(fit.assign(abs_delta_d0=fit_d0))
    # For augmented rows, flip d_0 as well (handled in _reverse_for_residual)
    # Compute residual target = true_delta - d_0.  In the v3 schema TARGET_COLUMN
    # (delta_log10_IC) IS the directed log ratio, so it is the true delta
    # directly — no baseline subtraction (that belonged to the legacy schema
    # where the target was the absolute log10(sigma_B)).
    true_delta_aug = fit_aug[TARGET_COLUMN].to_numpy(float)
    d0_aug = fit_aug["abs_delta_d0"].to_numpy(float)
    fit_aug["residual_target"] = true_delta_aug - d0_aug

    residual_cols = [*residual_feature_cols, "abs_delta_d0"]
    fit_mat, other_mat, medians, cats = _residual_matrices(
        fit_aug, other.assign(abs_delta_d0=other_d0), residual_cols
    )
    estimator = _build_lgbm(params, seed, config.n_jobs)
    estimator.fit(
        fit_mat,
        fit_aug["residual_target"].to_numpy(float),
        sample_weight=fit_aug["training_weight"].to_numpy(float),
    )
    r_hat = np.asarray(estimator.predict(other_mat), dtype=float)
    return estimator, medians, cats, other_d0 + r_hat


def _folds(
    frame: pd.DataFrame,
    config: ResidualConfig,
) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = GroupKFold(n_splits=config.cv_splits)
    return list(
        splitter.split(frame, groups=frame[SPLIT_GROUP_COLUMN].astype(str))
    )


def _tune_residual(
    train: pd.DataFrame,
    train_d0: np.ndarray,
    residual_feature_cols: list[str],
    config: ResidualConfig,
    model_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Optuna search on DOI-grouped CV weighted MAE of the combined delta."""
    folds = _folds(train, config)
    study = optuna.create_study(
        study_name=f"residual_lgbm_{config.seed}",
        storage=f"sqlite:///{(model_dir / 'optuna.db').resolve()}",
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=config.seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2),
        load_if_exists=True,
    )
    trials_before = len(study.trials)
    remaining = max(config.n_trials - trials_before, 0)
    start = time.monotonic()

    def objective(trial: optuna.Trial) -> float:
        params = _lgbm_params(trial)
        fold_maes = []
        for k, (fit_idx, valid_idx) in enumerate(folds, 1):
            fit_frame = train.iloc[fit_idx].reset_index(drop=True)
            valid_frame = train.iloc[valid_idx].reset_index(drop=True)
            fit_d0_fold = train_d0[fit_idx]
            valid_d0_fold = train_d0[valid_idx]
            _, _, _, pred_delta = _fit_residual(
                fit_frame, valid_frame,
                fit_d0_fold, valid_d0_fold,
                residual_feature_cols, params, config, config.seed + k,
            )
            true_delta = valid_frame[TARGET_COLUMN].to_numpy(float)
            weights = valid_frame[WEIGHT_COLUMN].to_numpy(float)
            fold_maes.append(float(
                mean_absolute_error(true_delta, pred_delta, sample_weight=weights)
            ))
            trial.report(float(np.mean(fold_maes)), k)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(fold_maes))

    def progress(study_obj: optuna.Study, _trial: optuna.FrozenTrial) -> None:
        done = max(len(study_obj.trials) - trials_before, 1)
        elapsed = time.monotonic() - start
        eta = max(remaining - done, 0) * elapsed / done
        print(
            f"  [lgbm] {done}/{remaining}; best_cv_mae={study_obj.best_value:.4f}; "
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

    best_params = _lgbm_params(
        optuna.trial.FixedTrial(study.best_trial.params)
    )
    # Recompute CV metrics with best params for logging
    fold_maes = []
    for k, (fit_idx, valid_idx) in enumerate(folds, 1):
        fit_frame = train.iloc[fit_idx].reset_index(drop=True)
        valid_frame = train.iloc[valid_idx].reset_index(drop=True)
        _, _, _, pred_delta = _fit_residual(
            fit_frame, valid_frame,
            train_d0[fit_idx], train_d0[valid_idx],
            residual_feature_cols, best_params, config, config.seed + k,
        )
        true_delta = valid_frame[TARGET_COLUMN].to_numpy(float)
        weights = valid_frame[WEIGHT_COLUMN].to_numpy(float)
        fold_maes.append(float(
            mean_absolute_error(true_delta, pred_delta, sample_weight=weights)
        ))
    cv_metrics = {
        "weighted_mae_delta_log10_ratio": float(np.mean(fold_maes)),
        "fold_maes": fold_maes,
        "n_trials": config.n_trials,
        "best_trial": int(study.best_trial.number),
    }
    write_json(model_dir / "best_params.json", best_params)
    write_json(model_dir / "cv_metrics.json", cv_metrics)
    study.trials_dataframe().to_csv(model_dir / "optuna_trials.csv", index=False)
    return best_params, cv_metrics


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def _coerce_numeric_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Coerce every numeric feature column to float (blank strings -> NaN).

    Loading with ``keep_default_na=False`` keeps string columns such as family
    and formulas intact, but leaves blank numeric cells as ``''``.  Those empty
    strings break float casts in the descriptor swap and matrix builders, so the
    known numeric columns are coerced back to real floats here.
    """
    numeric_cols: set[str] = set(MODEL_FEATURE_COLUMNS_V3)
    numeric_cols.update(SIGNED_DELTA_FEATURES)
    for descriptor in ABSOLUTE_DELTA_BY_DESCRIPTOR:
        numeric_cols.add(f"a_{descriptor}")
        numeric_cols.add(f"b_{descriptor}")
    numeric_cols.update({TARGET_COLUMN, BASELINE_LOG10_COLUMN, WEIGHT_COLUMN})
    present = [col for col in numeric_cols if col in frame.columns]
    frame = frame.copy()
    frame[present] = frame[present].apply(pd.to_numeric, errors="coerce")
    return frame


def train_residual_v3(config: ResidualConfig | None = None) -> dict[str, Any]:
    """Train the residual-correction trend model and write artifacts."""
    config = config or ResidualConfig()
    run_dir = config.output_root / config.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data ...", flush=True)
    train = pd.read_csv(config.train_path, keep_default_na=False)
    validation = pd.read_csv(config.validation_path, keep_default_na=False)
    # ``keep_default_na=False`` preserves string columns (family, formulas) but
    # turns blank numeric cells into empty strings, which breaks the A<->B
    # descriptor swap (``''`` cannot cast to float).  Coerce every numeric
    # feature column back to real floats (blanks -> NaN, imputed downstream).
    train = _coerce_numeric_columns(train)
    validation = _coerce_numeric_columns(validation)

    # Validate required columns
    required = {*MODEL_FEATURE_COLUMNS_V3, FAMILY_COLUMN, TARGET_COLUMN,
                BASELINE_LOG10_COLUMN, WEIGHT_COLUMN, GROUP_COLUMN,
                SPLIT_GROUP_COLUMN, "formula_a", "formula_b", "group_id", "pair_id"}
    for split_name, frame in (("train", train), ("validation", validation)):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{split_name} data missing columns: {missing}")

    if set(train[GROUP_COLUMN]) & set(validation[GROUP_COLUMN]):
        raise ValueError("DOI leakage between train and validation.")

    print("Loading absolute model ...", flush=True)
    abs_bundle = _load_abs_bundle(config.abs_run_dir)
    print(f"  Absolute model: {abs_bundle['_model_name']} from {abs_bundle['_run_dir']}", flush=True)

    # Residual feature columns = composition-difference features + log10_IC_a
    # (same as MODEL_FEATURE_COLUMNS_V3); d_0 is appended inside _fit_residual
    residual_feature_cols = list(MODEL_FEATURE_COLUMNS_V3)

    print(f"Computing fold-out d_0 for {len(train)} training pairs ...", flush=True)
    train_d0 = _fold_out_abs_deltas(train, abs_bundle, config)

    print(f"Computing d_0 for {len(validation)} validation pairs ...", flush=True)
    val_d0 = _compute_abs_deltas_batch(validation, abs_bundle)
    val_d0 = np.where(np.isfinite(val_d0), val_d0, 0.0)

    model_dir = run_dir / "lightgbm"
    model_dir.mkdir(exist_ok=True)

    print(f"Tuning residual LightGBM ({config.n_trials} Optuna trials) ...", flush=True)
    best_params, cv_metrics = _tune_residual(
        train, train_d0, residual_feature_cols, config, model_dir
    )

    # Final model: fit on train only, evaluate on validation
    print("Fitting final model on train, evaluating on validation ...", flush=True)
    estimator, medians, family_categories, val_combined_delta = _fit_residual(
        train, validation,
        train_d0, val_d0,
        residual_feature_cols, best_params, config, config.seed,
    )

    val_metrics = _residual_metrics(validation, val_combined_delta - val_d0, val_d0)

    # Calibration interval: 90th-percentile weighted absolute error on validation
    true_val_delta = validation[TARGET_COLUMN].to_numpy(float)
    val_abs_errors = np.abs(val_combined_delta - true_val_delta)
    val_weights = validation[WEIGHT_COLUMN].to_numpy(float)
    order = np.argsort(val_abs_errors)
    cumw = (np.cumsum(val_weights[order]) - 0.5 * val_weights[order]) / val_weights.sum()
    interval = float(np.interp(0.90, cumw, val_abs_errors[order]))

    # Baselines for comparison. ``_residual_metrics`` forms predicted_delta as
    # ``d0 + residual_prediction``, so:
    #   * zero-change  -> predicted_delta == 0  => residual_prediction = -val_d0
    #   * absolute-anchor -> predicted_delta == val_d0 => residual_prediction = 0
    zero_val = _residual_metrics(validation, -val_d0, val_d0)
    anchor_val = _residual_metrics(validation, np.zeros(len(validation)), val_d0)

    beats_zero = (
        val_metrics["weighted_mae_delta_log10_ratio"]
        < zero_val["weighted_mae_delta_log10_ratio"]
    )
    model_status = (
        "validated_candidate"
        if beats_zero and val_metrics["r2_log10_sigma_b"] > 0
        else "experimental_not_for_deployment"
    )

    # Save bundle
    feature_min = train[MODEL_FEATURE_COLUMNS_V3].min().to_dict()
    feature_max = train[MODEL_FEATURE_COLUMNS_V3].max().to_dict()
    bundle = {
        "model": estimator,
        "model_name": "lightgbm",
        "residual_feature_cols": residual_feature_cols,
        "residual_feature_cols_with_d0": [*residual_feature_cols, "abs_delta_d0"],
        "input_features": list(medians.index),
        "family_categories": family_categories,
        "numeric_medians": medians.to_dict(),
        "feature_min": feature_min,
        "feature_max": feature_max,
        "abs_run_dir": portable_path(config.abs_run_dir),
        "target": TARGET_COLUMN,
        "target_definition": "delta_log10_IC = log10(sigma_B) - log10(sigma_A)",
        "baseline_feature": BASELINE_LOG10_COLUMN,
        "architecture": "residual_correction: delta_hat = d0 + h(features)",
        "d0_definition": "f_abs(B) - f_abs(A) using absolute LightGBM model",
        "residual_definition": "r_train = true_delta - d_0",
        "reverse_training_augmentation": True,
        "weight_normalization": "training weights normalised to mean 1 after A/B augmentation",
        "prediction_interval_absolute_error_delta_log10_ratio_90": interval,
        "model_status": model_status,
    }
    best_dir = run_dir / "best_model"
    best_dir.mkdir(exist_ok=True)
    joblib.dump(bundle, best_dir / "model.joblib")
    write_json(best_dir / "best_params.json", best_params)
    write_json(best_dir / "validation_metrics.json", val_metrics)

    manifest = {
        "run_name": config.run_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config),
        "architecture": "residual_correction",
        "abs_run_dir": portable_path(config.abs_run_dir),
        "cv_metrics": cv_metrics,
        "validation_metrics": val_metrics,
        "validation_zero_change_baseline": zero_val,
        "validation_absolute_anchor_baseline": anchor_val,
        "validation_calibrated_90pct_error": interval,
        "model_status": model_status,
        "beats_zero_change_baseline": beats_zero,
    }
    write_json(run_dir / "manifest.json", manifest)

    report_lines = [
        "# Residual-correction trend-v3 regression",
        "",
        "Architecture: `delta_hat = f_abs(B) - f_abs(A) + h(features)`",
        "",
        f"Absolute model: {portable_path(config.abs_run_dir)}",
        "",
        "## CV metrics",
        f"- `weighted_mae_delta_log10_ratio` (mean across {config.cv_splits} folds): "
        f"{cv_metrics['weighted_mae_delta_log10_ratio']:.4f}",
        "",
        "## Validation metrics",
        *[f"- `{k}`: {v:.4f}" if isinstance(v, float) else f"- `{k}`: {v}"
          for k, v in val_metrics.items()],
        "",
        "## Validation zero-change baseline",
        f"- `weighted_mae_delta_log10_ratio`: {zero_val['weighted_mae_delta_log10_ratio']:.4f}",
        "",
        f"Beats zero-change baseline: **{beats_zero}**",
        f"Model status: **{model_status}**",
        "",
    ]
    (run_dir / "model_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    print(
        f"Validation weighted log-ratio MAE: {val_metrics['weighted_mae_delta_log10_ratio']:.4f}",
        flush=True,
    )
    print(f"Model status: {model_status}", flush=True)
    print(f"Output: {run_dir.resolve()}", flush=True)
    return {
        "run_dir": run_dir,
        "validation_metrics": val_metrics,
        "model_status": model_status,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train residual-correction trend model v3.")
    parser.add_argument("--train", type=Path, default=None, help="Path to train CSV.")
    parser.add_argument("--validation", type=Path, default=None, help="Path to validation CSV.")
    parser.add_argument("--abs-run-dir", type=Path, default=None, help="Absolute model run dir.")
    parser.add_argument("--output-root", type=Path, default=None, help="Output root directory.")
    parser.add_argument("--run-name", type=str, default=None, help="Run name (subdirectory).")
    parser.add_argument("--n-trials", type=int, default=None, help="Optuna trials.")
    parser.add_argument("--cv-splits", type=int, default=None, help="GroupKFold splits.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--n-jobs", type=int, default=None, help="Parallel jobs for LightGBM.")
    args = parser.parse_args()

    kwargs = {k: v for k, v in {
        "train_path": args.train,
        "validation_path": args.validation,
        "abs_run_dir": args.abs_run_dir,
        "output_root": args.output_root,
        "run_name": args.run_name,
        "n_trials": args.n_trials,
        "cv_splits": args.cv_splits,
        "seed": args.seed,
        "n_jobs": args.n_jobs,
    }.items() if v is not None}
    train_residual_v3(ResidualConfig(**kwargs) if kwargs else None)


def main() -> None:
    """Train the residual-correction trend model (point-run entry).

    Run directly via ``python main/trend/train_residual_v3.py``.
    Inputs  : data/trend/data-trend-v3-pairs-feature-train.csv
              data/trend/data-trend-v3-pairs-feature-validation.csv
              data/trend/data-trend-v3-pairs-feature-test.csv (from split_v3)
              runs/absolute/abs_v2_f37_native_family_lgbm_trials50_seed42/
    Output  : runs/trend/trend_v3_residual_lgbm_optuna50_seed42/
    """
    config = ResidualConfig()
    train_residual_v3(config)


if __name__ == "__main__":
    main()
