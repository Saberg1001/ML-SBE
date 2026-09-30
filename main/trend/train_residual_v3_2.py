"""Residual-correction trend model v3.2 — cross-fitted absolute anchor.

Identical to train_residual_v3 except that training-set d₀ values are computed
via *true* cross-fitting: for every DOI-grouped fold the absolute LightGBM model
is retrained from scratch on the remaining folds, then used to predict d₀ for
the held-out fold.  This eliminates the in-sample d₀ leakage present in v3,
where the fixed anchor model had seen every training material.

Deployment:  one final absolute model (trained on the full absolute dataset,
i.e. the same bundle used in v3) is still used at inference time.  Only the
*training-set* d₀ computation changes.

Cross-fitting scheme
---------------------
  trend_train DOIs  ──► GroupKFold(K=cv_splits)
  for each fold k:
      abs_fit   = absolute rows whose trend-DOI ∉ fold-k trend-DOIs
      abs_held  = absolute rows whose trend-DOI  ∈ fold-k trend-DOIs  (excluded)
      retrain lightgbm on abs_fit  ──► fold_abs_model_k
      d₀[fold_k_pairs] = fold_abs_model_k(B) − fold_abs_model_k(A)

The absolute training data is loaded from the same raw CSV used for abs_v2.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import spearmanr

from main.paths import DATA_DIR, RUNS_DIR
from main.features import FeatureConfig, make_feature_table, normalize_family
from main.absolute.train import TrainConfig as AbsTrainConfig, train_model as abs_train_model
from main.absolute.split import SplitConfig, split_feature_table
from main.trend.train_residual_v3 import (
    ResidualConfig,
    _abs_predict_single_formula,
    _augment_residual,
    _residual_matrices,
    _residual_metrics,
    _tune_residual,
    _fit_residual,
    _coerce_numeric_columns,
    FAMILY_COLUMN,
    TARGET_COLUMN,
    BASELINE_LOG10_COLUMN,
    WEIGHT_COLUMN,
    GROUP_COLUMN,
    SPLIT_GROUP_COLUMN,
    MODEL_FEATURE_COLUMNS_V3,
)
from main.trend.features_v3 import MODEL_FEATURE_COLUMNS_V3  # noqa: F811 (same constant)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Raw absolute-v2 source (same as used for abs_v2_f37 training)
ABS_SOURCE_CSV = DATA_DIR / "absolute" / "data-absolute-v2-model-clean.csv"

# Descriptor cache from the existing abs_v2 run (avoids recomputing descriptors)
ABS_DESCRIPTOR_CACHE = (
    RUNS_DIR / "absolute" / "abs_v2_f37_native_family_lgbm_trials50_seed42"
    / "data" / "all_features.csv"
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ResidualV32Config(ResidualConfig):
    """Extends ResidualConfig with cross-fitting parameters."""

    # Path to the absolute source CSV for cross-fit retraining
    abs_source_csv: Path = ABS_SOURCE_CSV
    # Optuna trials for each fold-k absolute model (fewer than full run is OK)
    abs_fold_n_trials: int = 20
    # run_name override; default auto-generated
    run_name: str = "trend_v3_2_residual_crossfit_f56_swap_groupcv5_optuna50_seed42"


# ---------------------------------------------------------------------------
# Absolute feature table (cached)
# ---------------------------------------------------------------------------

def _build_abs_feature_table(abs_source_csv: Path) -> pd.DataFrame:
    """Build (or load from cache) the full F37 feature table for abs data."""
    if ABS_DESCRIPTOR_CACHE.exists():
        print(f"  [abs-feat] loading cached descriptors from {ABS_DESCRIPTOR_CACHE}", flush=True)
        return pd.read_csv(ABS_DESCRIPTOR_CACHE)

    print("  [abs-feat] computing descriptors (no cache found) ...", flush=True)
    source = pd.read_csv(abs_source_csv)
    feat_cfg = FeatureConfig(
        min_conductivity=None,
        include_family=True,
        family_encoding="native",
        include_interactions=True,
        include_small_features=True,
        drop_redundant=True,
        output_path=None,
    )
    result = make_feature_table(source, feat_cfg)
    return result.table


# ---------------------------------------------------------------------------
# Retrain one fold-k absolute model
# ---------------------------------------------------------------------------

def _retrain_abs_fold(
    abs_feat: pd.DataFrame,
    exclude_dois: set[str],
    fold_k: int,
    tmp_dir: Path,
    config: ResidualV32Config,
    abs_feature_cols: list[str],
) -> dict[str, Any]:
    """Train an absolute model on abs_feat rows whose DOI ∉ exclude_dois.

    Uses the exact feature columns of the deployment anchor (incl. ``family``)
    so fold models are schema-compatible with ``_abs_predict_single_formula``.
    Returns the joblib bundle dict (same schema as abs_v2).
    """
    # Filter: keep rows not belonging to the held-out trend DOIs
    doi_col = "DOI" if "DOI" in abs_feat.columns else "doi"
    if doi_col in abs_feat.columns:
        mask = ~abs_feat[doi_col].isin(exclude_dois)
        fit_rows = abs_feat[mask].copy()
    else:
        # No DOI column in abs feature table → use all rows (conservative)
        fit_rows = abs_feat.copy()

    if len(fit_rows) < 50:
        raise ValueError(
            f"Fold {fold_k}: only {len(fit_rows)} absolute rows remain after "
            f"excluding {len(exclude_dois)} trend DOIs — too few to retrain."
        )

    # 80/20 random split (same ratio as original abs_v2 training)
    split_result = split_feature_table(
        fit_rows,
        SplitConfig(method="random", test_size=0.2, seed=config.seed + fold_k),
    )
    fold_run_name = f"_crossfit_fold{fold_k}_seed{config.seed}"
    fold_dir = tmp_dir / fold_run_name

    train_result = abs_train_model(
        split_result.train,
        split_result.test,
        AbsTrainConfig(
            model_name="lightgbm",
            n_trials=config.abs_fold_n_trials,
            cv_splits=3,               # faster inner CV for fold models
            seed=config.seed + fold_k,
            optuna_seed=config.seed + fold_k,
            output_root=tmp_dir,
            run_name=fold_run_name,
            dataset_name="abs_crossfit_fold",
            feature_columns=abs_feature_cols,  # exact v2 schema incl. family
            categorical_features=["family"],
            n_jobs=config.n_jobs,
            verbose=False,
        ),
    )
    bundle_path = fold_dir / "lightgbm" / "model.joblib"
    bundle = joblib.load(bundle_path)
    return bundle


# ---------------------------------------------------------------------------
# Cross-fitted d₀
# ---------------------------------------------------------------------------

def _crossfit_abs_deltas(
    train: pd.DataFrame,
    abs_feat: pd.DataFrame,
    config: ResidualV32Config,
    tmp_dir: Path,
    abs_feature_cols: list[str],
) -> np.ndarray:
    """Compute cross-fitted d₀ for every training pair.

    For each DOI-grouped fold k:
      1. Identify trend-train DOIs in fold k (held-out DOIs).
      2. Remove abs rows whose DOI overlaps fold-k trend DOIs.
      3. Retrain an absolute model on the remaining abs rows.
      4. Predict d₀ for fold-k trend pairs using the fold model.
    """
    splitter = GroupKFold(n_splits=config.cv_splits)
    d0 = np.full(len(train), math.nan, dtype=float)
    groups = train[SPLIT_GROUP_COLUMN].astype(str).to_numpy()

    total_start = time.monotonic()
    for fold_k, (fit_idx, held_idx) in enumerate(
        splitter.split(train, groups=groups), start=1
    ):
        held_frame = train.iloc[held_idx].reset_index(drop=True)
        held_dois: set[str] = set(held_frame[SPLIT_GROUP_COLUMN].astype(str))

        elapsed = time.monotonic() - total_start
        print(
            f"  [crossfit] fold {fold_k}/{config.cv_splits} "
            f"— held-out DOIs: {len(held_dois)}, pairs: {len(held_idx)} "
            f"(elapsed {elapsed/60:.1f}min)",
            flush=True,
        )

        fold_bundle = _retrain_abs_fold(
            abs_feat, held_dois, fold_k, tmp_dir, config, abs_feature_cols
        )

        # Predict d₀ for held-out pairs
        fold_d0 = []
        for _, row in held_frame.iterrows():
            try:
                fa = str(row["formula_a"])
                fb = str(row["formula_b"])
                fam = normalize_family(str(row.get(FAMILY_COLUMN, "unknown")))
                la = _abs_predict_single_formula(fa, fam, fold_bundle)
                lb = _abs_predict_single_formula(fb, fam, fold_bundle)
                fold_d0.append(lb - la)
            except Exception:
                fold_d0.append(math.nan)

        d0[held_idx] = np.array(fold_d0, dtype=float)

        n_fin = int(np.isfinite(d0[held_idx]).sum())
        print(
            f"  [crossfit] fold {fold_k} done — "
            f"finite d₀: {n_fin}/{len(held_idx)}",
            flush=True,
        )

    n_bad = int(np.sum(~np.isfinite(d0)))
    if n_bad:
        print(
            f"  [crossfit] {n_bad}/{len(d0)} non-finite d₀ → imputed with 0.0",
            flush=True,
        )
        d0 = np.where(np.isfinite(d0), d0, 0.0)
    return d0


# ---------------------------------------------------------------------------
# Helpers (reused from v3)
# ---------------------------------------------------------------------------

def _write_json(path: Path, obj: Any) -> None:
    import json
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def _portable_path(p: Path) -> str:
    try:
        return str(p.relative_to(Path.cwd()))
    except ValueError:
        return str(p)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def train_residual_v3_2(config: ResidualV32Config | None = None) -> dict[str, Any]:
    """Train the cross-fitted residual trend model (v3.2)."""
    config = config or ResidualV32Config()
    run_dir = config.output_root / config.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = run_dir / "_crossfit_tmp"
    tmp_dir.mkdir(exist_ok=True)

    print("Loading trend data ...", flush=True)
    train = pd.read_csv(config.train_path, keep_default_na=False)
    validation = pd.read_csv(config.validation_path, keep_default_na=False)
    train = _coerce_numeric_columns(train)
    validation = _coerce_numeric_columns(validation)

    required = {
        *MODEL_FEATURE_COLUMNS_V3, FAMILY_COLUMN, TARGET_COLUMN,
        BASELINE_LOG10_COLUMN, WEIGHT_COLUMN, GROUP_COLUMN,
        SPLIT_GROUP_COLUMN, "formula_a", "formula_b", "group_id", "pair_id",
    }
    for split_name, frame in (("train", train), ("validation", validation)):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{split_name} missing columns: {missing}")

    if set(train[GROUP_COLUMN]) & set(validation[GROUP_COLUMN]):
        raise ValueError("DOI leakage between train and validation.")

    print(
        f"Train: {len(train)} pairs / {train[SPLIT_GROUP_COLUMN].nunique()} DOIs",
        flush=True,
    )
    print(
        f"Validation: {len(validation)} pairs / {validation[SPLIT_GROUP_COLUMN].nunique()} DOIs",
        flush=True,
    )

    # ── Load deployment anchor (for val d₀ and final inference) ─────────────
    print("Loading deployment absolute bundle ...", flush=True)
    from main.trend.train_residual_v3 import _load_abs_bundle
    deploy_abs_bundle = _load_abs_bundle(config.abs_run_dir)
    print(
        f"  Deployment anchor: {deploy_abs_bundle['_model_name']} "
        f"from {deploy_abs_bundle['_run_dir']}",
        flush=True,
    )

    # ── Build absolute feature table for cross-fitting ───────────────────────
    print("Building absolute descriptor table for cross-fitting ...", flush=True)
    abs_feat = _build_abs_feature_table(config.abs_source_csv)
    print(f"  Absolute feature rows: {len(abs_feat)}", flush=True)

    # ── Cross-fitted training d₀ ──────────────────────────────────────────────
    abs_feature_cols = list(deploy_abs_bundle["feature_cols"])  # 37 cols incl. family
    print(
        f"Computing cross-fitted d₀ for {len(train)} training pairs "
        f"({config.cv_splits} folds × {config.abs_fold_n_trials} Optuna trials each) ...",
        flush=True,
    )
    train_d0 = _crossfit_abs_deltas(train, abs_feat, config, tmp_dir, abs_feature_cols)

    # ── Validation d₀ (deployment anchor, same as v3) ─────────────────────────
    print(f"Computing d₀ for {len(validation)} validation pairs ...", flush=True)
    from main.trend.train_residual_v3 import _compute_abs_deltas_batch
    val_d0 = _compute_abs_deltas_batch(validation, deploy_abs_bundle)
    val_d0 = np.where(np.isfinite(val_d0), val_d0, 0.0)

    # ── Residual feature columns ───────────────────────────────────────────────
    residual_feature_cols = list(MODEL_FEATURE_COLUMNS_V3)

    # ── Tune & fit (identical to v3 from here) ─────────────────────────────────
    model_dir = run_dir / "lightgbm"
    model_dir.mkdir(exist_ok=True)

    print(
        f"Tuning residual LightGBM ({config.n_trials} Optuna trials) ...",
        flush=True,
    )
    best_params, cv_metrics = _tune_residual(
        train, train_d0, residual_feature_cols, config, model_dir
    )

    print("Fitting final model on train, evaluating on validation ...", flush=True)
    estimator, medians, family_categories, val_combined_delta = _fit_residual(
        train, validation,
        train_d0, val_d0,
        residual_feature_cols, best_params, config, config.seed,
    )

    val_metrics = _residual_metrics(
        validation, val_combined_delta - val_d0, val_d0
    )

    # Calibration
    true_val_delta = validation[TARGET_COLUMN].to_numpy(float)
    val_abs_errors = np.abs(val_combined_delta - true_val_delta)
    val_weights = validation[WEIGHT_COLUMN].to_numpy(float)
    order = np.argsort(val_abs_errors)
    cumw = (
        np.cumsum(val_weights[order]) - 0.5 * val_weights[order]
    ) / val_weights.sum()
    interval = float(np.interp(0.90, cumw, val_abs_errors[order]))

    # Baselines
    zero_val   = _residual_metrics(validation, -val_d0, val_d0)
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

    # ── Save bundle ────────────────────────────────────────────────────────────
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
        "abs_run_dir": _portable_path(config.abs_run_dir),
        "target": TARGET_COLUMN,
        "target_definition": "delta_log10_IC = log10(sigma_B) - log10(sigma_A)",
        "baseline_feature": BASELINE_LOG10_COLUMN,
        "architecture": "residual_correction_v3_2: delta_hat = d0 + h(features)",
        "d0_definition": "f_abs(B) - f_abs(A) using deployment absolute model",
        "d0_training_method": "cross-fitted: fold-k abs model retrained without fold-k trend DOIs",
        "residual_definition": "r_train = true_delta - cross_fitted_d0",
        "reverse_training_augmentation": True,
        "prediction_interval_absolute_error_delta_log10_ratio_90": interval,
        "model_status": model_status,
        "version": "v3.2",
    }
    best_dir = run_dir / "best_model"
    best_dir.mkdir(exist_ok=True)
    joblib.dump(bundle, best_dir / "model.joblib")
    _write_json(best_dir / "best_params.json", best_params)
    _write_json(best_dir / "validation_metrics.json", val_metrics)

    manifest = {
        "run_name": config.run_name,
        "version": "v3.2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(config).items()},
        "architecture": "residual_correction_v3_2",
        "d0_training": "cross-fitted (fold-k abs model excludes fold-k trend DOIs)",
        "abs_run_dir": _portable_path(config.abs_run_dir),
        "cv_metrics": cv_metrics,
        "validation_metrics": val_metrics,
        "validation_zero_change_baseline": zero_val,
        "validation_absolute_anchor_baseline": anchor_val,
        "validation_calibrated_90pct_error": interval,
        "model_status": model_status,
        "beats_zero_change_baseline": beats_zero,
    }
    _write_json(run_dir / "manifest.json", manifest)

    print(
        f"\nValidation weighted log-ratio MAE : "
        f"{val_metrics['weighted_mae_delta_log10_ratio']:.4f}",
        flush=True,
    )
    print(f"Direction accuracy (val)          : "
          f"{val_metrics['direction_accuracy_at_0.1_log10']:.4f}", flush=True)
    print(f"Model status                       : {model_status}", flush=True)
    print(f"Output                             : {run_dir}", flush=True)

    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train residual-v3.2 cross-fitted trend model"
    )
    parser.add_argument("--train",       default=str(DATA_DIR / "trend" / "data-trend-v3-pairs-feature-train.csv"))
    parser.add_argument("--validation",  default=str(DATA_DIR / "trend" / "data-trend-v3-pairs-feature-validation.csv"))
    parser.add_argument("--abs-run-dir", default=str(RUNS_DIR / "absolute" / "abs_v2_f37_native_family_lgbm_trials50_seed42"))
    parser.add_argument("--abs-source",  default=str(ABS_SOURCE_CSV))
    parser.add_argument("--n-trials",    type=int, default=50)
    parser.add_argument("--abs-fold-trials", type=int, default=20)
    parser.add_argument("--cv-splits",   type=int, default=5)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--n-jobs",      type=int, default=4)
    parser.add_argument("--run-name",    default="trend_v3_2_residual_crossfit_f56_swap_groupcv5_optuna50_seed42")
    args = parser.parse_args()

    cfg = ResidualV32Config(
        train_path=Path(args.train),
        validation_path=Path(args.validation),
        abs_run_dir=Path(args.abs_run_dir),
        abs_source_csv=Path(args.abs_source),
        n_trials=args.n_trials,
        abs_fold_n_trials=args.abs_fold_trials,
        cv_splits=args.cv_splits,
        seed=args.seed,
        n_jobs=args.n_jobs,
        run_name=args.run_name,
        output_root=RUNS_DIR / "trend",
    )
    train_residual_v3_2(cfg)
