"""Unified evaluation for trend-v3 regression models.

Answers one question: does any candidate model actually beat the trivial
zero-change baseline on the held-out split, under a single consistent
metric definition?

Why this exists
---------------
The per-run `validation_metrics.json` files report weighted MAE but no
baseline, so a model that degenerates to a constant prediction can post an
apparently respectable MAE.  The `trend_v3_reg_f55_*` run is the motivating
case: it predicts exactly 0.0 for every validation row (prediction std == 0)
while still reporting `weighted_mae = 0.27`.

This module therefore:

* evaluates every candidate on the *same* validation rows with the *same*
  metric code path;
* always reports the zero-change baseline alongside, and the skill
  difference against it;
* groups pairs by DOI and removes duplicate formula pairs within a DOI
  before computing group-level ranking metrics;
* marks degenerate models by name rather than emitting a bare boolean.

Metric convention
-----------------
`delta` is the directed log10 ratio ``log10(sigma_B) - log10(sigma_A)``.
All MAE / RMSE / R2 numbers are in log10 units.  `delta_log10_ratio`.

Entry point
-----------
    python main/trend/evaluate_unified.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

if __package__ is None:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from main.features import normalize_family
from main.paths import DATA_DIR, RUNS_DIR
from main.trend.features_v3 import (
    MODEL_FEATURE_COLUMNS_V3,
    TARGET_COLUMN,  # "delta_log10_IC"
)
from main.trend.regression_v3 import (
    BASELINE_LOG10_COLUMN,  # "log10_conductivity_a" — used by train_pair_regression_v3
    WEIGHT_COLUMN,
    write_json,
)
from main.trend.split_v3 import DEFAULT_TRAIN_V3, DEFAULT_VALIDATION_V3

FAMILY_COLUMN = "family"
GROUP_COLUMN = "doi"
# In the v3 feature CSVs the target IS the log-ratio (delta_log10_IC = log10_IC_b - log10_IC_a).
LOG_RATIO_COLUMN = TARGET_COLUMN   # "delta_log10_IC"
BASELINE_IC_COLUMN = "log10_IC_a"  # the baseline feature in the data
NEUTRAL_DELTA = 0.1  # |delta| below this counts as "unchanged" in log10 units

DEFAULT_OUTPUT_JSON = RUNS_DIR / "trend" / "unified_evaluation.json"
DEFAULT_OUTPUT_MD = Path("reports") / "trend" / "trend_v3_unified_evaluation.md"

# The direct trend model used as the reference candidate.
DEFAULT_DIRECT_RUN = (
    RUNS_DIR / "trend" / "trend_v3_reg_f56_swprefix_swap_groupcv5_optuna50_seed42"
)
DEFAULT_RESIDUAL_RUN = (
    RUNS_DIR / "trend" / "trend_v3_residual_f56_swap_groupcv5_optuna50_seed42"
)


# ---------------------------------------------------------------------------
# Candidate definitions
# ---------------------------------------------------------------------------

@dataclass
class Candidate:
    """One thing that can be scored on the held-out split.

    `predictor` returns `(predicted_delta, non_finite_anchor_ratio)`.  The
    ratio is always 0.0 for models that do not depend on an absolute-model
    anchor.
    """

    name: str
    kind: str  # "model" | "baseline"
    run_dir: Path | None = None
    model_dir: Path | None = None
    predictor: Callable[[pd.DataFrame], tuple[np.ndarray, float]] | None = None
    notes: str = ""
    # For anchor-based models (residual): the absolute-model run whose training
    # formulas may overlap the trend validation set (a leakage channel).
    anchor_run_dir: Path | None = None


# ---------------------------------------------------------------------------
# Bundle loading / prediction
# ---------------------------------------------------------------------------

def _resolve_model_dir(run_dir: Path) -> Path:
    """Return the directory holding the selected model's model.joblib."""
    best_txt = run_dir / "best_model.txt"
    if best_txt.exists():
        name = best_txt.read_text(encoding="utf-8").strip()
        candidate = run_dir / name
        if (candidate / "model.joblib").exists():
            return candidate
    if (run_dir / "best_model" / "model.joblib").exists():
        return run_dir / "best_model"
    raise FileNotFoundError(f"No model.joblib found under {run_dir}")


def _load_direct_bundle(model_dir: Path) -> dict[str, Any]:
    """Load a direct trend-v3 bundle, backfilling fields the trainer omitted.

    The f55 trainer writes ``family_categories`` into ``preprocessing.json``
    rather than into the joblib bundle, so the native-categorical family
    levels are recovered from there.  Without them a native-category booster
    cannot be fed at inference time.
    """
    bundle = joblib.load(model_dir / "model.joblib")
    if not bundle.get("family_categories"):
        preprocessing_path = model_dir / "preprocessing.json"
        if preprocessing_path.exists():
            try:
                preprocessing = json.loads(preprocessing_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                preprocessing = {}
            categories = preprocessing.get("family_categories") or []
            if categories:
                bundle["family_categories"] = [str(c) for c in categories]
            bundle.setdefault(
                "categorical_features", preprocessing.get("categorical_features", [])
            )
    if not bundle.get("family_categories"):
        # Last resort: a LightGBM booster trained on a ``pd.Categorical`` family
        # column stores the level ordering in ``booster_.pandas_categorical``.
        # This is the authoritative training-time mapping when no sidecar exists.
        family_column = str(bundle.get("family_column", FAMILY_COLUMN))
        booster = getattr(bundle.get("model"), "booster_", None)
        pandas_categorical = getattr(booster, "pandas_categorical", None)
        if (
            booster is not None
            and pandas_categorical
            and booster.feature_name()
            and booster.feature_name()[-1] == family_column
        ):
            bundle["family_categories"] = [
                str(value) for value in pandas_categorical[-1]
            ]
            bundle.setdefault("categorical_features", [family_column])
    return bundle


def _direct_matrix(frame: pd.DataFrame, bundle: dict[str, Any]) -> pd.DataFrame | np.ndarray:
    """Build the design matrix for a direct trend-v3 bundle.

    Two family encodings are supported, decided by the bundle contents:

    * native categorical — family is passed as a ``pd.Categorical`` column
      matching the training categories.  The result must stay a DataFrame,
      because converting to a numpy array would drop the category dtype that
      LightGBM relies on to identify the categorical feature;
    * one-hot — a ``family_categories`` list in the bundle defines the
      column block (unseen families map to an all-zero vector).

    When the bundle carries neither, family is not a model input at all.
    """
    model = bundle["model"]
    numeric_features = list(bundle.get("numeric_features", MODEL_FEATURE_COLUMNS_V3))
    medians = pd.Series(bundle.get("numeric_medians", {}), dtype=float)
    numeric = (
        frame[numeric_features]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(medians.reindex(numeric_features).fillna(0.0))
        .astype(float)
    )

    family = (
        frame.get(FAMILY_COLUMN, pd.Series("unknown", index=frame.index))
        .map(normalize_family)
        .astype(str)
    )

    categorical_features = list(bundle.get("categorical_features", []) or [])
    categories = [str(value) for value in bundle.get("family_categories", []) or []]

    native_family = (
        FAMILY_COLUMN in categorical_features
        or _model_uses_native_family(model, bundle)
    ) and FAMILY_COLUMN not in _one_hot_columns(bundle)

    if native_family and categories:
        # Match the training-time categories; unseen families become NaN,
        # which LightGBM routes to the default branch (the same treatment
        # `pd.Categorical` gives them during training).
        matrix = numeric.copy()
        matrix[FAMILY_COLUMN] = pd.Categorical(
            family.where(family.isin(categories)), categories=categories
        )
        return matrix

    if categories:
        # One-hot block, columns in bundle order
        one_hot = pd.get_dummies(family).reindex(columns=categories, fill_value=0)
        return np.concatenate(
            [numeric.to_numpy(np.float32), one_hot.to_numpy(np.float32)], axis=1
        )

    return numeric.to_numpy(np.float32)


def _one_hot_columns(bundle: dict[str, Any]) -> set[str]:
    """Detection helper: does the bundle describe an explicit one-hot block?"""
    return set(bundle.get("one_hot_family_columns", []) or [])


def _model_uses_native_family(model: Any, bundle: dict[str, Any]) -> bool:
    """True when the fitted booster expects a raw ``family`` column."""
    family_column = str(bundle.get("family_column", FAMILY_COLUMN))
    booster = getattr(model, "booster_", None)
    if booster is None:
        return False
    try:
        return booster.feature_name()[-1] == family_column
    except Exception:
        return False


def predict_direct(
    frame: pd.DataFrame,
    bundle: dict[str, Any],
) -> np.ndarray:
    """Predict delta_log10_IC directly (or reconstruct it from log10_sigma_b bundles).

    The f55 trainer predicts ``delta_log10_IC`` directly.  Older bundles that
    predict ``log10_conductivity_b`` are handled by subtracting the baseline
    feature.
    """
    predicted = np.asarray(
        bundle["model"].predict(_direct_matrix(frame, bundle)), dtype=float
    )
    target_col = str(bundle.get("target_column", ""))
    if target_col in ("delta_log10_IC", "delta_log10_conductivity"):
        # Model already outputs a delta — return as-is
        return predicted
    # Legacy: model outputs absolute log10(sigma_B); subtract the baseline
    baseline_key = BASELINE_IC_COLUMN  # "log10_IC_a"
    if baseline_key not in frame.columns:
        baseline_key = BASELINE_LOG10_COLUMN  # "log10_conductivity_a" fallback
    baseline = frame[baseline_key].to_numpy(float)
    return predicted - baseline


def predict_residual(
    frame: pd.DataFrame,
    bundle: dict[str, Any],
    abs_bundle: dict[str, Any],
) -> tuple[np.ndarray, float]:
    """delta_hat = d_0 + h(features), where d_0 = f_abs(B) - f_abs(A).

    Returns the predicted delta and the fraction of rows whose absolute-model
    anchor d_0 could not be computed (non-finite).  A high ratio means the
    residual model is effectively being asked to predict from features alone.
    """
    from main.trend.train_residual_v3 import _compute_abs_deltas_batch

    feature_cols = list(bundle["residual_feature_cols_with_d0"])
    d0 = _compute_abs_deltas_batch(frame, abs_bundle)
    non_finite_ratio = float(np.mean(~np.isfinite(d0)))
    d0 = np.where(np.isfinite(d0), d0, 0.0)

    prepared = frame.assign(abs_delta_d0=d0)
    medians = pd.Series(bundle["numeric_medians"], dtype=float)

    # Numeric block: coerce, clean, and impute against the bundle's stored
    # fit-time medians so the training scaling is reproduced exactly.
    numeric = (
        prepared[feature_cols]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(medians.reindex(feature_cols).fillna(0.0))
        .to_numpy(np.float32)
    )

    # One-hot family block, built against the TRAINING category order stored in
    # the bundle (not re-derived from this frame, whose family set may be a
    # subset — that would yield too few columns and a feature-count mismatch).
    categories = [str(value) for value in bundle.get("family_categories", []) or []]
    if categories:
        family = frame.get(FAMILY_COLUMN, pd.Series("unknown", index=frame.index))
        family = family.map(normalize_family).astype(str)
        one_hot = (
            pd.get_dummies(family)
            .reindex(columns=categories, fill_value=0)
            .to_numpy(np.float32)
        )
        matrix = np.concatenate([numeric, one_hot], axis=1)
    else:
        matrix = numeric

    residual = np.asarray(bundle["model"].predict(matrix), dtype=float)
    return d0 + residual, non_finite_ratio


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def predict_zero_change(frame: pd.DataFrame) -> tuple[np.ndarray, float]:
    """Predict no change at all.  The bar every model must clear."""
    return np.zeros(len(frame), dtype=float), 0.0


def predict_group_mean_change(frame: pd.DataFrame) -> tuple[np.ndarray, float]:
    """Predict the mean delta of the row's own DOI group.

    Diagnostic only: it is not a deployable predictor, because it needs the
    observed deltas of the group it is predicting.
    """
    values = (
        frame.groupby("group_id")[LOG_RATIO_COLUMN]
        .transform("mean")
        .to_numpy(float)
    )
    return values, 0.0


# ---------------------------------------------------------------------------
# Duplicate-formula handling within a DOI
# ---------------------------------------------------------------------------

def _pair_key(row: pd.Series) -> str:
    """Unordered, composition-normalised key for a pair."""
    values = sorted(
        {str(row["formula_a"]).strip().lower(), str(row["formula_b"]).strip().lower()}
    )
    return "|".join(values)


def deduplicate_pairs_within_doi(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop repeated formula pairs that occur more than once in the same DOI.

    A DOI can report the same two compositions twice (different samples or
    conditions).  Keeping both double-counts one comparison and silently
    reweights the group, so whole duplicated pairs are dropped.

    Returns the deduplicated frame and the number of dropped rows.
    """
    keyed = frame.assign(_pair_key=frame.apply(_pair_key, axis=1))
    duplicated = keyed.duplicated([GROUP_COLUMN, "_pair_key"], keep=False)
    dropped = int(duplicated.sum())
    kept = keyed.loc[~duplicated].drop(columns="_pair_key").reset_index(drop=True)
    return kept, dropped


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def direction_accuracy(true_delta: np.ndarray, predicted_delta: np.ndarray) -> float:
    """Sign agreement, with |delta| < 0.1 treated as its own "unchanged" class."""
    true_class = np.where(np.abs(true_delta) < NEUTRAL_DELTA, 0.0, np.sign(true_delta))
    pred_class = np.where(
        np.abs(predicted_delta) < NEUTRAL_DELTA, 0.0, np.sign(predicted_delta)
    )
    return float(np.mean(true_class == pred_class))


def group_top1_accuracy(frame: pd.DataFrame, predicted_delta: np.ndarray) -> float:
    """Within each DOI group, does the top-ranked pair hold the best delta?

    Ranking is done on the delta itself (not log10 sigma_B), because the
    downstream question is "which modification helps most", not "which
    material is most conductive".
    """
    scored = frame.assign(_pred=predicted_delta)
    hits: list[float] = []
    for _, group in scored.groupby("group_id"):
        if len(group) < 2:
            continue
        best_pred_row = group.iloc[int(np.argmax(group["_pred"].to_numpy(float)))]
        hits.append(float(best_pred_row[LOG_RATIO_COLUMN] == group[LOG_RATIO_COLUMN].max()))
    return float(np.mean(hits)) if hits else math.nan


def group_absolute_top1_accuracy(
    frame: pd.DataFrame,
    predicted_delta: np.ndarray,
) -> float:
    """Within each group, rank by final absolute ``sigma_B``.

    The model predicts a change delta, so the predicted final level is
    ``log10(sigma_A) + predicted_delta``.  The reference ranking uses the
    measured ``log10(sigma_B)`` (equivalently baseline + true delta).  Each
    group contributes one hit and groups with fewer than two pairs are
    skipped, matching :func:`group_top1_accuracy`.
    """
    scored = frame.assign(
        _pred_level=frame[BASELINE_IC_COLUMN].to_numpy(float)
        + np.asarray(predicted_delta, dtype=float),
        _true_level=frame[BASELINE_IC_COLUMN].to_numpy(float)
        + frame[LOG_RATIO_COLUMN].to_numpy(float),
    )
    hits: list[float] = []
    for _, group in scored.groupby("group_id"):
        if len(group) < 2:
            continue
        pred_idx = int(np.argmax(group["_pred_level"].to_numpy(float)))
        true_idx = int(np.argmax(group["_true_level"].to_numpy(float)))
        hits.append(float(pred_idx == true_idx))
    return float(np.mean(hits)) if hits else math.nan


def group_top_fraction_accuracy(
    frame: pd.DataFrame,
    predicted_scores: np.ndarray,
    true_scores: np.ndarray,
    fraction: float = 0.25,
) -> float:
    """Fraction of groups whose true-best row is in the predicted top fraction.

    ``ceil(fraction * group_size)`` candidates are retained per group, with at
    least one candidate.  Groups contribute equally, and groups with fewer
    than two rows are skipped.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1].")
    scored = frame.assign(
        _pred_score=np.asarray(predicted_scores, dtype=float),
        _true_score=np.asarray(true_scores, dtype=float),
    )
    hits: list[float] = []
    for _, group in scored.groupby("group_id"):
        n = len(group)
        if n < 2:
            continue
        k = max(1, int(math.ceil(fraction * n)))
        pred_order = np.argsort(-group["_pred_score"].to_numpy(float), kind="stable")
        true_best = int(np.argmax(group["_true_score"].to_numpy(float)))
        hits.append(float(true_best in pred_order[:k]))
    return float(np.mean(hits)) if hits else math.nan


def evaluate_predictions(
    frame: pd.DataFrame,
    true_delta: np.ndarray,
    predicted_delta: np.ndarray,
) -> dict[str, Any]:
    """All metrics for one set of predictions on one frame.

    In v3 data: ``delta_log10_IC = log10_IC_b - log10_IC_a``, so
    ``log10_IC_b = log10_IC_a + true_delta`` (used for R² of the target level).
    """
    weights = frame[WEIGHT_COLUMN].to_numpy(float)
    error = true_delta - predicted_delta
    baseline = frame[BASELINE_IC_COLUMN].to_numpy(float)

    increase_true = true_delta > NEUTRAL_DELTA
    increase_pred = predicted_delta > NEUTRAL_DELTA
    decrease_true = true_delta < -NEUTRAL_DELTA
    decrease_pred = predicted_delta < -NEUTRAL_DELTA

    spearman = (
        float(spearmanr(true_delta, predicted_delta).statistic)
        if np.std(true_delta) > 0 and np.std(predicted_delta) > 0
        else math.nan
    )

    # R² of log10(sigma_B): compare baseline + predicted_delta vs baseline + true_delta
    true_level = baseline + true_delta
    pred_level = baseline + predicted_delta
    r2_level = float(
        r2_score(true_level, pred_level, sample_weight=weights)
    ) if np.std(true_level) > 0 else math.nan

    return {
        "n_pairs": int(len(frame)),
        "n_dois": int(frame[GROUP_COLUMN].nunique()),
        "weighted_mae_delta_log10_ratio": float(
            mean_absolute_error(true_delta, predicted_delta, sample_weight=weights)
        ),
        "mae_delta_log10_ratio": float(mean_absolute_error(true_delta, predicted_delta)),
        "weighted_rmse_delta_log10_ratio": float(
            math.sqrt(mean_squared_error(true_delta, predicted_delta, sample_weight=weights))
        ),
        "rmse_delta_log10_ratio": float(
            math.sqrt(mean_squared_error(true_delta, predicted_delta))
        ),
        "weighted_r2_log10_sigma_b": r2_level,
        "r2_delta_log10_ratio": float(
            r2_score(true_delta, predicted_delta, sample_weight=weights)
        ) if np.std(true_delta) > 0 else math.nan,
        "spearman_delta_log10_ratio": spearman,
        "direction_accuracy_at_0.1_log10": direction_accuracy(true_delta, predicted_delta),
        "increase_precision_at_0.1_log10": float(
            np.sum(increase_true & increase_pred) / max(np.sum(increase_pred), 1)
        ),
        "increase_recall_at_0.1_log10": float(
            np.sum(increase_true & increase_pred) / max(np.sum(increase_true), 1)
        ),
        "decrease_precision_at_0.1_log10": float(
            np.sum(decrease_true & decrease_pred) / max(np.sum(decrease_pred), 1)
        ),
        "decrease_recall_at_0.1_log10": float(
            np.sum(decrease_true & decrease_pred) / max(np.sum(decrease_true), 1)
        ),
        "within_two_fold_ratio": float(np.mean(np.abs(error) <= math.log10(2.0))),
        "within_five_fold_ratio": float(np.mean(np.abs(error) <= math.log10(5.0))),
        "median_absolute_error_delta_log10_ratio": float(np.median(np.abs(error))),
        "p90_absolute_error_delta_log10_ratio": float(np.quantile(np.abs(error), 0.9)),
        "prediction_std_delta_log10_ratio": float(np.std(predicted_delta)),
        "group_top1_accuracy": group_top1_accuracy(frame, predicted_delta),
        "group_absolute_top1_accuracy": group_absolute_top1_accuracy(
            frame, predicted_delta
        ),
        "group_delta_top25_accuracy": group_top_fraction_accuracy(
            frame, predicted_delta, true_delta, fraction=0.25
        ),
        "group_absolute_top25_accuracy": group_top_fraction_accuracy(
            frame,
            baseline + predicted_delta,
            baseline + true_delta,
            fraction=0.25,
        ),
    }


# ---------------------------------------------------------------------------
# Degenerate-model detection
# ---------------------------------------------------------------------------

def degenerate_reasons(
    metrics: dict[str, Any],
    zero_metrics: dict[str, Any],
    non_finite_d0_ratio: float = 0.0,
    eps: float = 1e-9,
) -> list[str]:
    """Name every reason this model must not be deployed.

    An empty list means the model cleared all structural checks; it does not
    by itself mean the model is useful.
    """
    reasons: list[str] = []
    if metrics["prediction_std_delta_log10_ratio"] <= eps:
        reasons.append("constant_prediction (zero variance across held-out rows)")
    if (
        metrics["weighted_mae_delta_log10_ratio"]
        >= zero_metrics["weighted_mae_delta_log10_ratio"]
    ):
        reasons.append("no_skill_vs_zero_change (weighted MAE not better than predicting no change)")
    if metrics["r2_delta_log10_ratio"] < 0:
        reasons.append("negative_r2 (explains less delta variance than the mean predictor)")
    if non_finite_d0_ratio > 0.30:
        reasons.append(
            f"absolute_anchor_missing ({non_finite_d0_ratio:.0%} of rows had non-finite d_0)"
        )
    return reasons


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _load_abs_bundle_for_residual(residual_bundle: dict[str, Any]) -> dict[str, Any]:
    from main.trend.train_residual_v3 import _load_abs_bundle

    return _load_abs_bundle(Path(residual_bundle["abs_run_dir"]))


def build_candidates(
    direct_run: Path,
    residual_run: Path | None,
) -> list[Candidate]:
    candidates: list[Candidate] = []

    if direct_run.exists():
        model_dir = _resolve_model_dir(direct_run)
        bundle = _load_direct_bundle(model_dir)
        candidates.append(
            Candidate(
                name="direct_trend_v3",
                kind="model",
                run_dir=direct_run,
                model_dir=model_dir,
                predictor=lambda frame, b=bundle: (predict_direct(frame, b), 0.0),
                notes=f"{bundle.get('model_name', 'unknown')} direct delta regression",
            )
        )

    if residual_run is not None and residual_run.exists() and (
        residual_run / "best_model" / "model.joblib"
    ).exists():
        model_dir = residual_run / "best_model"
        bundle = joblib.load(model_dir / "model.joblib")
        abs_bundle = _load_abs_bundle_for_residual(bundle)
        candidates.append(
            Candidate(
                name="residual_trend_v3",
                kind="model",
                run_dir=residual_run,
                model_dir=model_dir,
                predictor=lambda frame, b=bundle, a=abs_bundle: predict_residual(frame, b, a),
                notes="delta_hat = d_0 + h(features)",
                anchor_run_dir=_resolve_anchor_run_dir(bundle.get("abs_run_dir")),
            )
        )

    candidates.append(
        Candidate(
            name="zero_change",
            kind="baseline",
            predictor=predict_zero_change,
            notes="predict delta = 0 (no change)",
        )
    )
    candidates.append(
        Candidate(
            name="group_mean_change",
            kind="baseline",
            predictor=predict_group_mean_change,
            notes="diagnostic only: uses the group's own observed deltas",
        )
    )
    return candidates


def _resolve_anchor_run_dir(raw: str | None) -> Path | None:
    """Resolve a portable (possibly project-relative) anchor run path."""
    if not raw:
        return None
    from main.paths import PROJECT_ROOT
    candidate = Path(raw)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    # Try project-relative resolution
    project_candidate = PROJECT_ROOT / candidate
    if project_candidate.exists():
        return project_candidate
    return candidate if candidate.exists() else None


def _normalize_formula(value: Any) -> str:
    """Whitespace-insensitive formula key for overlap comparison."""
    return str(value).strip().replace(" ", "")


def _anchor_training_formulas(anchor_run_dir: Path) -> set[str]:
    """Normalized formulas the absolute anchor model was fit/evaluated on.

    Reads the anchor run's persisted train (and test) split.  These are the
    compositions whose measured conductivity the anchor has effectively seen;
    a trend-validation pair with an endpoint here gets a d_0 that recalls a
    training label rather than a genuine out-of-sample prediction.
    """
    seen: set[str] = set()
    for split in ("train.csv", "test.csv"):
        path = anchor_run_dir / "data" / split
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        for col in ("True Composition", "Reduced Composition", "formula"):
            if col in frame.columns:
                seen.update(frame[col].map(_normalize_formula))
                break
    return seen


def anchor_leakage_diagnostic(
    frame: pd.DataFrame,
    anchor_run_dir: Path,
) -> dict[str, Any]:
    """Fraction of validation pairs whose endpoints the anchor model has seen.

    A high ``both_endpoints`` fraction means the residual anchor d_0 is largely
    recalling training labels, so the residual model's held-out metrics
    overstate genuine deployment performance.
    """
    seen = _anchor_training_formulas(anchor_run_dir)
    if not seen or "formula_a" not in frame.columns or "formula_b" not in frame.columns:
        return {"available": False}
    a_seen = frame["formula_a"].map(_normalize_formula).isin(seen).to_numpy()
    b_seen = frame["formula_b"].map(_normalize_formula).isin(seen).to_numpy()
    n = len(frame)
    return {
        "available": True,
        "anchor_run_dir": str(anchor_run_dir),
        "n_pairs": int(n),
        "anchor_training_formulas": int(len(seen)),
        "either_endpoint_seen_ratio": float(np.mean(a_seen | b_seen)),
        "both_endpoints_seen_ratio": float(np.mean(a_seen & b_seen)),
        "neither_endpoint_seen_pairs": int(np.sum(~(a_seen | b_seen))),
    }


def run_unified_evaluation(
    train_path: Path = DEFAULT_TRAIN_V3,
    validation_path: Path = DEFAULT_VALIDATION_V3,
    direct_run: Path = DEFAULT_DIRECT_RUN,
    residual_run: Path | None = DEFAULT_RESIDUAL_RUN,
    output_json: Path = DEFAULT_OUTPUT_JSON,
    output_md: Path = DEFAULT_OUTPUT_MD,
) -> dict[str, Any]:
    validation = pd.read_csv(validation_path, keep_default_na=False)
    train = pd.read_csv(train_path, keep_default_na=False)

    overlap = set(train[GROUP_COLUMN]) & set(validation[GROUP_COLUMN])
    if overlap:
        raise ValueError(f"DOI leakage between train and validation: {sorted(overlap)[:5]}")

    deduped, dropped = deduplicate_pairs_within_doi(validation)
    print(
        f"Validation: {len(validation)} rows, {validation[GROUP_COLUMN].nunique()} DOIs; "
        f"after within-DOI dedup: {len(deduped)} rows ({dropped} duplicate rows dropped)",
        flush=True,
    )

    true_delta = deduped[LOG_RATIO_COLUMN].to_numpy(float)
    candidates = build_candidates(direct_run, residual_run)

    results: dict[str, Any] = {}
    non_finite_by_candidate: dict[str, float] = {}
    for candidate in candidates:
        assert candidate.predictor is not None
        predicted, non_finite_ratio = candidate.predictor(deduped)
        non_finite_by_candidate[candidate.name] = non_finite_ratio
        metrics = evaluate_predictions(deduped, true_delta, predicted)
        results[candidate.name] = {
            "kind": candidate.kind,
            "notes": candidate.notes,
            "run_dir": str(candidate.run_dir) if candidate.run_dir else None,
            "metrics": metrics,
        }
        print(
            f"  {candidate.name:20s} weighted_mae={metrics['weighted_mae_delta_log10_ratio']:.4f} "
            f"dir_acc={metrics['direction_accuracy_at_0.1_log10']:.3f} "
            f"delta_top1={metrics['group_top1_accuracy']:.3f} "
            f"absolute_top1={metrics['group_absolute_top1_accuracy']:.3f} "
            f"delta_top25={metrics['group_delta_top25_accuracy']:.3f} "
            f"absolute_top25={metrics['group_absolute_top25_accuracy']:.3f}",
            flush=True,
        )

    zero_metrics = results["zero_change"]["metrics"]

    # Attach baselines/skill and degeneracy verdicts
    candidates_by_name = {candidate.name: candidate for candidate in candidates}
    for name, entry in results.items():
        candidate = candidates_by_name[name]
        metrics = entry["metrics"]
        entry["skill_vs_zero_change"] = {
            "weighted_mae_improvement": float(
                zero_metrics["weighted_mae_delta_log10_ratio"]
                - metrics["weighted_mae_delta_log10_ratio"]
            ),
            "weighted_mae_ratio": float(
                metrics["weighted_mae_delta_log10_ratio"]
                / max(zero_metrics["weighted_mae_delta_log10_ratio"], 1e-12)
            ),
            "beats_zero_change": bool(
                metrics["weighted_mae_delta_log10_ratio"]
                < zero_metrics["weighted_mae_delta_log10_ratio"]
            ),
        }
        if entry["kind"] == "model":
            reasons = degenerate_reasons(
                metrics,
                zero_metrics,
                non_finite_by_candidate.get(name, 0.0),
            )
            # Anchor-leakage check: a residual model whose absolute anchor was
            # fitted on (many of) the validation endpoints has an optimistic
            # d_0, so its skill is not an out-of-sample deployment estimate.
            if candidate.anchor_run_dir is not None:
                leakage = anchor_leakage_diagnostic(deduped, candidate.anchor_run_dir)
                entry["anchor_leakage"] = leakage
                if leakage.get("available") and (
                    leakage["either_endpoint_seen_ratio"] > 0.10
                ):
                    reasons.append(
                        "anchor_leakage_suspected "
                        f"({leakage['either_endpoint_seen_ratio']:.0%} of pairs have an "
                        "endpoint in the anchor model's training data)"
                    )
            entry["degenerate_reasons"] = reasons
            entry["model_status"] = (
                "experimental_not_for_deployment" if reasons else "validated_candidate"
            )

    deployable = [
        name
        for name, entry in results.items()
        if entry["kind"] == "model" and not entry.get("degenerate_reasons")
    ]
    verdict = {
        "deployable_models": deployable,
        "summary": (
            f"{len(deployable)} model(s) cleared all structural checks: {deployable}"
            if deployable
            else "No model cleared the structural checks; keep the zero-change baseline."
        ),
    }

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "validation_path": str(validation_path),
        "train_path": str(train_path),
        "metric_convention": "delta = log10(sigma_B) - log10(sigma_A); all errors in log10 units",
        "duplicate_handling": (
            "formula pairs occurring more than once within the same DOI are dropped whole"
        ),
        "rows_before_dedup": int(len(validation)),
        "rows_after_dedup": int(len(deduped)),
        "duplicate_rows_dropped": dropped,
        "results": results,
        "verdict": verdict,
    }

    write_json(output_json, payload)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(render_markdown(payload), encoding="utf-8")
    print(f"\nWrote {output_json}\nWrote {output_md}", flush=True)
    print(verdict["summary"], flush=True)
    return payload


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return "n/a" if not math.isfinite(value) else f"{value:.4f}"
    return str(value)


def render_markdown(payload: dict[str, Any]) -> str:
    results = payload["results"]
    lines = [
        "# Trend-v3 unified evaluation",
        "",
        f"Generated: {payload['generated_at_utc']}",
        "",
        f"Metric convention: `{payload['metric_convention']}`",
        "",
        f"Held-out split: `{payload['validation_path']}`",
        "",
        f"Rows: {payload['rows_before_dedup']} before within-DOI dedup, "
        f"{payload['rows_after_dedup']} after "
        f"({payload['duplicate_rows_dropped']} duplicate pair rows dropped).",
        "",
        "## Headline comparison",
        "",
        "| candidate | kind | status | weighted MAE | vs zero-change | direction acc | within 2x | delta top-1 | absolute top-1 | delta top-25% | absolute top-25% |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    status_label = {
        "validated_candidate": "✅ validated",
        "experimental_not_for_deployment": "⚠️ flagged",
    }
    for name, entry in results.items():
        metrics = entry["metrics"]
        skill = entry["skill_vs_zero_change"]
        status = status_label.get(entry.get("model_status", ""), "—")
        lines.append(
            f"| `{name}` | {entry['kind']} | {status} | "
            f"{_fmt(metrics['weighted_mae_delta_log10_ratio'])} | "
            f"{'**beats**' if skill['beats_zero_change'] else 'no'} "
            f"({skill['weighted_mae_improvement']:+.4f}) | "
            f"{_fmt(metrics['direction_accuracy_at_0.1_log10'])} | "
            f"{_fmt(metrics['within_two_fold_ratio'])} | "
            f"{_fmt(metrics['group_top1_accuracy'])} | "
            f"{_fmt(metrics['group_absolute_top1_accuracy'])} | "
            f"{_fmt(metrics['group_delta_top25_accuracy'])} | "
            f"{_fmt(metrics['group_absolute_top25_accuracy'])} |"
        )

    lines += [
        "",
        "`vs zero-change` is `zero_change_MAE - model_MAE`; positive is better. "
        "A ⚠️ flagged model's held-out numbers are not a trustworthy deployment "
        "estimate — see its degeneracy check below.",
        "",
        "## Degeneracy checks",
        "",
    ]

    for name, entry in results.items():
        if entry["kind"] != "model":
            continue
        reasons = entry.get("degenerate_reasons", [])
        lines.append(f"### `{name}`")
        lines.append("")
        lines.append(f"Status: **{entry.get('model_status')}**")
        lines.append("")
        if reasons:
            lines.append("Failing conditions:")
            lines += [f"- {reason}" for reason in reasons]
        else:
            lines.append("Cleared all structural checks.")
        lines.append("")
        leakage = entry.get("anchor_leakage")
        if leakage and leakage.get("available"):
            lines += [
                "Absolute-anchor leakage check:",
                f"- either endpoint in anchor training data: "
                f"{leakage['either_endpoint_seen_ratio']:.1%} of pairs",
                f"- both endpoints in anchor training data: "
                f"{leakage['both_endpoints_seen_ratio']:.1%} of pairs",
                f"- pairs with neither endpoint seen (leak-free): "
                f"{leakage['neither_endpoint_seen_pairs']} of {leakage['n_pairs']}",
                "",
                "> A high overlap means d_0 recalls anchor training labels, so the "
                "held-out metrics above overstate genuine out-of-sample skill. An "
                "honest estimate needs an anchor retrained without the trend-"
                "validation materials.",
                "",
            ]

    lines += [
        "## Full metrics",
        "",
        "```json",
        json.dumps({k: v["metrics"] for k, v in results.items()}, indent=2),
        "```",
        "",
        "## Verdict",
        "",
        payload["verdict"]["summary"],
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Unified trend-v3 evaluation.")
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN_V3)
    parser.add_argument("--validation", type=Path, default=DEFAULT_VALIDATION_V3)
    parser.add_argument("--direct-run", type=Path, default=DEFAULT_DIRECT_RUN)
    parser.add_argument("--residual-run", type=Path, default=DEFAULT_RESIDUAL_RUN)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    args = parser.parse_args()

    run_unified_evaluation(
        train_path=args.train,
        validation_path=args.validation,
        direct_run=args.direct_run,
        residual_run=args.residual_run,
        output_json=args.output_json,
        output_md=args.output_md,
    )


if __name__ == "__main__":
    main()
