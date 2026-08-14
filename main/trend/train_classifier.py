"""Train trend classifiers with DOI-grouped Optuna tuning."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Allow running this module directly as a script (VS Code "Run" button): set
# __package__ so the relative imports below resolve, and expose project root.
if __package__ is None:
    _FILE = Path(__file__).resolve()
    if str(_FILE.parents[2]) not in sys.path:
        sys.path.insert(0, str(_FILE.parents[2]))
    __package__ = f"{_FILE.parents[1].name}.{_FILE.parents[0].name}"

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import shutil
import time
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import OneHotEncoder

from ..paths import RUNS_DIR, portable_path
from .features import (
    ABSOLUTE_DELTA_BY_DESCRIPTOR,
    MODEL_FEATURE_COLUMNS,
    SIGNED_DELTA_FEATURES,
    TREND_ABSOLUTE_THRESHOLD_S_CM,
)
from .split import DEFAULT_TRAIN, DEFAULT_VALIDATION


DEFAULT_OUTPUT_ROOT = RUNS_DIR / "trend"
MODEL_NAMES = ("lightgbm", "catboost", "xgboost", "random_forest")
FAMILY_COLUMN = "family"
TARGET_COLUMN = "trend_label"
WEIGHT_COLUMN = "pair_weight_group_equal"
GROUP_COLUMN = "doi"
CLASS_LABELS = ("decrease", "unchanged", "increase")
LABEL_TO_INDEX = {label: index for index, label in enumerate(CLASS_LABELS)}
INDEX_TO_LABEL = {index: label for label, index in LABEL_TO_INDEX.items()}
REVERSE_INDEX = np.array([2, 1, 0], dtype=int)


@dataclass(frozen=True)
class ClassifierConfig:
    train_path: Path = DEFAULT_TRAIN
    validation_path: Path = DEFAULT_VALIDATION
    output_root: Path = DEFAULT_OUTPUT_ROOT
    run_name: str | None = None
    models: tuple[str, ...] = MODEL_NAMES
    n_trials: int = 50
    cv_splits: int = 5
    seed: int = 42
    swap_augmentation: bool = True


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return portable_path(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dependency_versions() -> dict[str, str]:
    versions = {}
    for package in (
        "pandas",
        "numpy",
        "scikit-learn",
        "optuna",
        "lightgbm",
        "catboost",
        "xgboost",
    ):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "missing"
    return versions


def _validate_dependencies(models: tuple[str, ...]) -> None:
    packages = {
        "lightgbm": "lightgbm",
        "catboost": "catboost",
        "xgboost": "xgboost",
        "random_forest": "scikit-learn",
    }
    unsupported = sorted(set(models) - set(packages))
    if unsupported:
        raise ValueError(f"Unsupported models: {unsupported}")
    missing = []
    for model_name in models:
        try:
            metadata.version(packages[model_name])
        except metadata.PackageNotFoundError:
            missing.append(packages[model_name])
    if missing:
        raise RuntimeError(
            "Missing packages in the active environment: "
            + ", ".join(sorted(set(missing)))
        )


def _load_data(config: ClassifierConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(config.train_path, keep_default_na=False)
    validation = pd.read_csv(config.validation_path, keep_default_na=False)
    required = {*MODEL_FEATURE_COLUMNS, FAMILY_COLUMN, TARGET_COLUMN,
                WEIGHT_COLUMN, GROUP_COLUMN, "pair_id", "group_id"}
    for name, frame in (("train", train), ("validation", validation)):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{name} data is missing columns: {missing}")
        unknown = sorted(set(frame[TARGET_COLUMN]) - set(CLASS_LABELS))
        if unknown:
            raise ValueError(f"{name} data has unsupported labels: {unknown}")
        numeric = frame[MODEL_FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
        if np.isinf(numeric.to_numpy(dtype=float)).any():
            raise ValueError(f"{name} data contains infinite model features.")
        frame[MODEL_FEATURE_COLUMNS] = numeric
        weights = pd.to_numeric(frame[WEIGHT_COLUMN], errors="coerce")
        if weights.isna().any() or (weights <= 0).any():
            raise ValueError(f"{name} data contains invalid pair weights.")
    if set(train[GROUP_COLUMN]) & set(validation[GROUP_COLUMN]):
        raise ValueError("DOI leakage detected between train and validation.")
    if set(train["pair_id"]) & set(validation["pair_id"]):
        raise ValueError("pair_id leakage detected between train and validation.")
    return train, validation


def _encode_labels(frame: pd.DataFrame) -> np.ndarray:
    return frame[TARGET_COLUMN].map(LABEL_TO_INDEX).to_numpy(dtype=int)


def _weights(frame: pd.DataFrame) -> np.ndarray:
    return pd.to_numeric(frame[WEIGHT_COLUMN], errors="raise").to_numpy(dtype=float)


def _reverse_frame(frame: pd.DataFrame) -> pd.DataFrame:
    reverse = frame.copy()
    for descriptor, delta_column in ABSOLUTE_DELTA_BY_DESCRIPTOR.items():
        reverse[f"a_{descriptor}"] = frame[f"b_{descriptor}"].to_numpy(dtype=float)
        reverse[f"b_{descriptor}"] = frame[f"a_{descriptor}"].to_numpy(dtype=float)
    reverse.loc[:, SIGNED_DELTA_FEATURES] = -reverse[
        SIGNED_DELTA_FEATURES
    ].to_numpy(dtype=float)
    reverse[TARGET_COLUMN] = reverse[TARGET_COLUMN].map(
        {
            "decrease": "increase",
            "unchanged": "unchanged",
            "increase": "decrease",
        }
    )
    return reverse


def _augment_training(frame: pd.DataFrame, enabled: bool) -> pd.DataFrame:
    forward = frame.copy()
    forward["training_weight"] = _weights(forward)
    forward["augmentation_direction"] = "forward"
    if not enabled:
        return forward
    reverse = _reverse_frame(frame)
    forward["training_weight"] *= 0.5
    reverse["training_weight"] = _weights(reverse) * 0.5
    reverse["augmentation_direction"] = "reverse"
    return pd.concat([forward, reverse], ignore_index=True)


def _one_hot_encoder() -> OneHotEncoder:
    return OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32)


def _prepare_native_family(
    train: pd.DataFrame,
    other: pd.DataFrame,
    model_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    columns = [*MODEL_FEATURE_COLUMNS, FAMILY_COLUMN]
    train_x = train.loc[:, columns].copy()
    other_x = other.loc[:, columns].copy()
    train_numeric = train_x[MODEL_FEATURE_COLUMNS].astype(float)
    other_numeric = other_x[MODEL_FEATURE_COLUMNS].astype(float)
    medians = train_numeric.median().fillna(0.0)
    train_x[MODEL_FEATURE_COLUMNS] = train_numeric.fillna(medians)
    other_x[MODEL_FEATURE_COLUMNS] = other_numeric.fillna(medians)
    categories = sorted(train_x[FAMILY_COLUMN].astype(str).unique())
    if model_name == "lightgbm":
        category_dtype = pd.CategoricalDtype(categories=categories)
        train_x[FAMILY_COLUMN] = train_x[FAMILY_COLUMN].astype(category_dtype)
        other_x[FAMILY_COLUMN] = other_x[FAMILY_COLUMN].astype(category_dtype)
    else:
        train_x[FAMILY_COLUMN] = train_x[FAMILY_COLUMN].astype(str)
        other_x[FAMILY_COLUMN] = other_x[FAMILY_COLUMN].astype(str)
    return train_x, other_x, {
        "family_categories": categories,
        "encoder": None,
        "numeric_medians": medians.to_dict(),
    }


def _prepare_one_hot(
    train: pd.DataFrame,
    other: pd.DataFrame,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, dict[str, Any]]:
    encoder = _one_hot_encoder()
    train_family = encoder.fit_transform(train[[FAMILY_COLUMN]].astype(str))
    other_family = encoder.transform(other[[FAMILY_COLUMN]].astype(str))
    train_numeric_frame = train[MODEL_FEATURE_COLUMNS].astype(float)
    other_numeric_frame = other[MODEL_FEATURE_COLUMNS].astype(float)
    medians = train_numeric_frame.median().fillna(0.0)
    train_numeric = sparse.csr_matrix(
        train_numeric_frame.fillna(medians).to_numpy(dtype=np.float32)
    )
    other_numeric = sparse.csr_matrix(
        other_numeric_frame.fillna(medians).to_numpy(dtype=np.float32)
    )
    train_x = sparse.hstack([train_numeric, train_family], format="csr")
    other_x = sparse.hstack([other_numeric, other_family], format="csr")
    names = [
        *MODEL_FEATURE_COLUMNS,
        *encoder.get_feature_names_out([FAMILY_COLUMN]).tolist(),
    ]
    return train_x, other_x, {
        "family_categories": encoder.categories_[0].astype(str).tolist(),
        "encoder": encoder,
        "encoded_feature_names": names,
        "numeric_medians": medians.to_dict(),
    }


def _prepare_matrices(
    train: pd.DataFrame,
    other: pd.DataFrame,
    model_name: str,
) -> tuple[Any, Any, dict[str, Any]]:
    if model_name in {"lightgbm", "catboost"}:
        return _prepare_native_family(train, other, model_name)
    return _prepare_one_hot(train, other)


def _sample_parameters(trial: optuna.Trial, model_name: str) -> dict[str, Any]:
    if model_name == "lightgbm":
        max_depth = trial.suggest_categorical("max_depth", [-1, 3, 4, 5, 6, 8, 10])
        max_leaves = 63 if max_depth == -1 else min(63, 2**max_depth)
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 600, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 7, max_leaves),
            "max_depth": max_depth,
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 60),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
        }
    if model_name == "catboost":
        return {
            "iterations": trial.suggest_int("iterations", 100, 600, step=100),
            "depth": trial.suggest_int("depth", 4, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 30.0, log=True),
            "random_strength": trial.suggest_float("random_strength", 1e-3, 10.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 5.0),
        }
    if model_name == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 600, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "min_child_weight": trial.suggest_float("min_child_weight", 0.5, 20.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 1e-8, 5.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
        }
    if model_name == "random_forest":
        max_features_kind = trial.suggest_categorical(
            "max_features_kind", ["sqrt", "log2", "float"]
        )
        max_features: str | float
        if max_features_kind == "float":
            max_features = trial.suggest_float("max_features_float", 0.3, 1.0)
        else:
            max_features = max_features_kind
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 700, step=100),
            "max_depth": trial.suggest_categorical(
                "max_depth", [None, 6, 8, 10, 15, 20, 30]
            ),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": max_features,
        }
    raise ValueError(f"Unsupported model: {model_name}")


def _build_model(model_name: str, params: dict[str, Any], seed: int) -> Any:
    if model_name == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            objective="multiclass",
            num_class=len(CLASS_LABELS),
            random_state=seed,
            n_jobs=4,
            verbosity=-1,
            subsample_freq=1,
            **params,
        )
    if model_name == "catboost":
        from catboost import CatBoostClassifier

        return CatBoostClassifier(
            loss_function="MultiClass",
            random_seed=seed,
            thread_count=4,
            verbose=False,
            allow_writing_files=False,
            **params,
        )
    if model_name == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            objective="multi:softprob",
            num_class=len(CLASS_LABELS),
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=seed,
            n_jobs=4,
            **params,
        )
    if model_name == "random_forest":
        return RandomForestClassifier(random_state=seed, n_jobs=4, **params)
    raise ValueError(f"Unsupported model: {model_name}")


def _fit_model(
    model: Any,
    model_name: str,
    train_x: Any,
    train_y: np.ndarray,
    train_weight: np.ndarray,
) -> None:
    kwargs: dict[str, Any] = {"sample_weight": train_weight}
    if model_name == "catboost":
        kwargs["cat_features"] = [FAMILY_COLUMN]
    elif model_name == "lightgbm":
        kwargs["categorical_feature"] = [FAMILY_COLUMN]
    model.fit(train_x, train_y, **kwargs)


def _predict_probabilities(model: Any, matrix: Any) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(matrix), dtype=float)
    if probabilities.shape[1] != len(CLASS_LABELS):
        raise ValueError(
            f"Expected {len(CLASS_LABELS)} probability columns, got "
            f"{probabilities.shape[1]}."
        )
    row_sums = probabilities.sum(axis=1, keepdims=True)
    if not np.isfinite(probabilities).all() or (row_sums <= 0).any():
        raise ValueError("Model emitted invalid class probabilities.")
    return probabilities / row_sums


def _fit_predict_symmetric(
    model_name: str,
    params: dict[str, Any],
    fit_frame: pd.DataFrame,
    predict_frame: pd.DataFrame,
    seed: int,
) -> tuple[Any, dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    augmented = _augment_training(fit_frame, enabled=True)
    reverse_predict = _reverse_frame(predict_frame)
    both_predict = pd.concat([predict_frame, reverse_predict], ignore_index=True)
    train_x, predict_x, preprocessing = _prepare_matrices(
        augmented, both_predict, model_name
    )
    model = _build_model(model_name, params, seed)
    _fit_model(
        model,
        model_name,
        train_x,
        _encode_labels(augmented),
        augmented["training_weight"].to_numpy(dtype=float),
    )
    all_probabilities = _predict_probabilities(model, predict_x)
    split_at = len(predict_frame)
    forward = all_probabilities[:split_at]
    reverse_aligned = all_probabilities[split_at:][:, REVERSE_INDEX]
    symmetric = (forward + reverse_aligned) / 2.0
    return model, preprocessing, symmetric, forward, reverse_aligned


def _fit_predict(
    model_name: str,
    params: dict[str, Any],
    fit_frame: pd.DataFrame,
    predict_frame: pd.DataFrame,
    seed: int,
    swap_augmentation: bool,
) -> tuple[Any, dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    if swap_augmentation:
        return _fit_predict_symmetric(
            model_name, params, fit_frame, predict_frame, seed
        )
    augmented = _augment_training(fit_frame, enabled=False)
    train_x, predict_x, preprocessing = _prepare_matrices(
        augmented, predict_frame, model_name
    )
    model = _build_model(model_name, params, seed)
    _fit_model(
        model,
        model_name,
        train_x,
        _encode_labels(augmented),
        augmented["training_weight"].to_numpy(dtype=float),
    )
    probabilities = _predict_probabilities(model, predict_x)
    return model, preprocessing, probabilities, probabilities, probabilities


def _doi_macro_f1(
    frame: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: np.ndarray,
) -> float:
    scores = []
    groups = frame[GROUP_COLUMN].astype(str).to_numpy()
    for doi in sorted(set(groups)):
        mask = groups == doi
        scores.append(
            f1_score(
                y_true[mask],
                y_pred[mask],
                labels=np.arange(len(CLASS_LABELS)),
                average="macro",
                sample_weight=sample_weight[mask],
                zero_division=0,
            )
        )
    return float(np.mean(scores))


def classification_metrics(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
) -> dict[str, Any]:
    y_true = _encode_labels(frame)
    y_pred = probabilities.argmax(axis=1)
    weights = _weights(frame)
    precision, recall, class_f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=np.arange(len(CLASS_LABELS)),
        sample_weight=weights,
        zero_division=0,
    )
    severe_mask = ((y_true == 0) & (y_pred == 2)) | (
        (y_true == 2) & (y_pred == 0)
    )
    return {
        "weighted_macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=np.arange(len(CLASS_LABELS)),
                average="macro",
                sample_weight=weights,
                zero_division=0,
            )
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=np.arange(len(CLASS_LABELS)),
                average="macro",
                zero_division=0,
            )
        ),
        "doi_macro_f1": _doi_macro_f1(frame, y_true, y_pred, weights),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "weighted_log_loss": float(
            log_loss(
                y_true,
                probabilities,
                labels=np.arange(len(CLASS_LABELS)),
                sample_weight=weights,
            )
        ),
        "weighted_ordinal_mae": float(
            np.average(np.abs(y_true - y_pred), weights=weights)
        ),
        "weighted_severe_reversal_rate": float(
            np.average(severe_mask.astype(float), weights=weights)
        ),
        "per_class": {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(class_f1[index]),
                "weighted_support": float(support[index]),
            }
            for index, label in enumerate(CLASS_LABELS)
        },
        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
            labels=np.arange(len(CLASS_LABELS)),
        ).tolist(),
        "weighted_confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
            labels=np.arange(len(CLASS_LABELS)),
            sample_weight=weights,
        ).tolist(),
    }


def _grouped_folds(
    frame: pd.DataFrame,
    config: ClassifierConfig,
) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = StratifiedGroupKFold(
        n_splits=config.cv_splits,
        shuffle=True,
        random_state=config.seed,
    )
    folds = list(
        splitter.split(
            frame,
            y=_encode_labels(frame),
            groups=frame[GROUP_COLUMN].astype(str),
        )
    )
    for fold_index, (fit_index, valid_index) in enumerate(folds, start=1):
        fit_dois = set(frame.iloc[fit_index][GROUP_COLUMN].astype(str))
        valid_dois = set(frame.iloc[valid_index][GROUP_COLUMN].astype(str))
        if fit_dois & valid_dois:
            raise ValueError(f"DOI leakage detected in CV fold {fold_index}.")
        if set(_encode_labels(frame.iloc[fit_index])) != set(range(len(CLASS_LABELS))):
            raise ValueError(f"CV fit fold {fold_index} is missing at least one class.")
        if set(_encode_labels(frame.iloc[valid_index])) != set(range(len(CLASS_LABELS))):
            raise ValueError(
                f"CV validation fold {fold_index} is missing at least one class."
            )
    return folds


def _cross_validate(
    model_name: str,
    params: dict[str, Any],
    train: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    config: ClassifierConfig,
    trial: optuna.Trial | None = None,
) -> tuple[float, list[dict[str, Any]]]:
    fold_results = []
    for fold_index, (fit_index, valid_index) in enumerate(folds, start=1):
        fit_frame = train.iloc[fit_index].reset_index(drop=True)
        valid_frame = train.iloc[valid_index].reset_index(drop=True)
        _, _, probabilities, forward, reverse_aligned = _fit_predict(
            model_name,
            params,
            fit_frame,
            valid_frame,
            config.seed + fold_index,
            config.swap_augmentation,
        )
        scores = classification_metrics(valid_frame, probabilities)
        scores["fold"] = fold_index
        scores["fit_rows_original"] = len(fit_frame)
        scores["validation_rows"] = len(valid_frame)
        scores["fit_dois"] = int(fit_frame[GROUP_COLUMN].nunique())
        scores["validation_dois"] = int(valid_frame[GROUP_COLUMN].nunique())
        scores["forward_reverse_probability_mae"] = float(
            np.mean(np.abs(forward - reverse_aligned))
        )
        fold_results.append(scores)
        running_score = float(
            np.mean([item["weighted_macro_f1"] for item in fold_results])
        )
        if trial is not None:
            trial.report(running_score, step=fold_index)
            if trial.should_prune():
                raise optuna.TrialPruned()
    score = float(np.mean([item["weighted_macro_f1"] for item in fold_results]))
    return score, fold_results


def _summarize_cv(fold_results: list[dict[str, Any]]) -> dict[str, Any]:
    metric_names = (
        "weighted_macro_f1",
        "macro_f1",
        "doi_macro_f1",
        "balanced_accuracy",
        "accuracy",
        "weighted_log_loss",
        "weighted_ordinal_mae",
        "weighted_severe_reversal_rate",
        "forward_reverse_probability_mae",
    )
    return {
        "mean_metrics": {
            name: float(np.mean([fold[name] for fold in fold_results]))
            for name in metric_names
        },
        "std_metrics": {
            name: float(np.std([fold[name] for fold in fold_results], ddof=0))
            for name in metric_names
        },
    }


def _study_trials_frame(study: optuna.Study) -> pd.DataFrame:
    rows = []
    for trial in study.trials:
        row = {
            "trial": trial.number,
            "state": trial.state.name,
            "value": trial.value,
            "duration_seconds": (
                trial.duration.total_seconds() if trial.duration is not None else None
            ),
        }
        row.update({f"param_{key}": value for key, value in trial.params.items()})
        row.update({f"user_{key}": value for key, value in trial.user_attrs.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def _optimize(
    model_name: str,
    train: pd.DataFrame,
    config: ClassifierConfig,
    model_dir: Path,
) -> tuple[optuna.Study, dict[str, Any], dict[str, Any]]:
    folds = _grouped_folds(train, config)
    storage_path = model_dir / "optuna.db"
    study = optuna.create_study(
        study_name=(
            f"trend_{model_name}_groupcv{config.cv_splits}_{config.seed}"
        ),
        storage=f"sqlite:///{storage_path.resolve()}",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=config.seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2),
        load_if_exists=True,
    )
    completed_before = sum(
        trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials
    )
    remaining_trials = max(config.n_trials - completed_before, 0)
    start = time.monotonic()

    def objective(trial: optuna.Trial) -> float:
        params = _sample_parameters(trial, model_name)
        score, fold_results = _cross_validate(
            model_name, params, train, folds, config, trial
        )
        cv_summary = _summarize_cv(fold_results)
        trial.set_user_attr(
            "cv_mean_doi_macro_f1",
            cv_summary["mean_metrics"]["doi_macro_f1"],
        )
        trial.set_user_attr(
            "cv_std_weighted_macro_f1",
            cv_summary["std_metrics"]["weighted_macro_f1"],
        )
        trial.set_user_attr(
            "fold_weighted_macro_f1",
            [fold["weighted_macro_f1"] for fold in fold_results],
        )
        return score

    def callback(study_object: optuna.Study, trial: optuna.FrozenTrial) -> None:
        elapsed = time.monotonic() - start
        finished_now = max(len(study_object.trials) - completed_before, 1)
        eta = max(remaining_trials - finished_now, 0) * elapsed / finished_now
        best = study_object.best_value
        print(
            f"[{model_name}] trial={trial.number + 1}; "
            f"best_grouped_cv_weighted_macro_f1={best:.4f}; "
            f"elapsed={elapsed / 60:.1f}min; eta={eta / 60:.1f}min",
            flush=True,
        )

    if remaining_trials:
        study.optimize(
            objective,
            n_trials=remaining_trials,
            callbacks=[callback],
            show_progress_bar=False,
        )
    if not any(
        trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials
    ):
        raise RuntimeError(f"No completed Optuna trial for {model_name}.")
    best_params = _sample_parameters(
        optuna.trial.FixedTrial(study.best_trial.params), model_name
    )
    best_score, best_fold_results = _cross_validate(
        model_name, best_params, train, folds, config
    )
    cv_summary = _summarize_cv(best_fold_results)
    cv_metrics = {
        "objective": (
            f"{config.cv_splits}-fold StratifiedGroupKFold DOI-grouped "
            "weighted Macro-F1"
        ),
        "group_column": GROUP_COLUMN,
        "n_splits": config.cv_splits,
        "best_value": best_score,
        **cv_summary,
        "folds": best_fold_results,
    }
    _study_trials_frame(study).to_csv(model_dir / "optuna_trials.csv", index=False)
    _write_json(model_dir / "best_params.json", best_params)
    _write_json(model_dir / "cv_metrics.json", cv_metrics)
    _write_json(model_dir / "tuning_metrics.json", cv_metrics)
    return study, best_params, cv_metrics


def _prediction_frame(
    source: pd.DataFrame,
    probabilities: np.ndarray,
    forward: np.ndarray,
    reverse_aligned: np.ndarray,
) -> pd.DataFrame:
    result = source.loc[
        :,
        [
            "pair_id",
            "group_id",
            "doi",
            "化学式_a",
            "化学式_b",
            FAMILY_COLUMN,
            TARGET_COLUMN,
            WEIGHT_COLUMN,
        ],
    ].copy()
    predicted_index = probabilities.argmax(axis=1)
    result["predicted_label"] = [INDEX_TO_LABEL[index] for index in predicted_index]
    result["correct"] = result[TARGET_COLUMN].eq(result["predicted_label"])
    for index, label in enumerate(CLASS_LABELS):
        result[f"probability_{label}"] = probabilities[:, index]
        result[f"forward_probability_{label}"] = forward[:, index]
        result[f"reverse_aligned_probability_{label}"] = reverse_aligned[:, index]
    result["forward_reverse_probability_mae"] = np.mean(
        np.abs(forward - reverse_aligned), axis=1
    )
    return result


def _save_feature_importance(
    model: Any,
    preprocessing: dict[str, Any],
    model_dir: Path,
) -> None:
    if not hasattr(model, "feature_importances_"):
        return
    importance = np.asarray(model.feature_importances_, dtype=float).reshape(-1)
    names = preprocessing.get("encoded_feature_names")
    if names is None:
        names = [*MODEL_FEATURE_COLUMNS, FAMILY_COLUMN]
    if len(names) != len(importance):
        return
    table = pd.DataFrame({"feature": names, "importance": importance})
    table.sort_values("importance", ascending=False).to_csv(
        model_dir / "feature_importance.csv", index=False
    )


def _train_model(
    model_name: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    config: ClassifierConfig,
    run_dir: Path,
) -> dict[str, Any]:
    model_dir = run_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    study, best_params, tuning_metrics = _optimize(
        model_name, train, config, model_dir
    )
    model, preprocessing, probabilities, forward, reverse_aligned = _fit_predict(
        model_name,
        best_params,
        train.reset_index(drop=True),
        validation.reset_index(drop=True),
        config.seed,
        config.swap_augmentation,
    )
    validation_metrics = classification_metrics(validation, probabilities)
    prediction_table = _prediction_frame(
        validation, probabilities, forward, reverse_aligned
    )
    prediction_table.to_csv(model_dir / "validation_predictions.csv", index=False)
    _write_json(model_dir / "validation_metrics.json", validation_metrics)
    _write_json(
        model_dir / "preprocessing.json",
        {
            "numeric_features": MODEL_FEATURE_COLUMNS,
            "categorical_features": [FAMILY_COLUMN],
            "family_categories": preprocessing["family_categories"],
            "encoded_feature_names": preprocessing.get("encoded_feature_names"),
            "swap_augmentation": config.swap_augmentation,
            "signed_features_reversed": SIGNED_DELTA_FEATURES,
            "endpoint_reverse_rule": "A_reverse = B_forward; B_reverse = A_forward",
            "numeric_medians": preprocessing["numeric_medians"],
        },
    )
    joblib.dump(
        {
            "model": model,
            "encoder": preprocessing.get("encoder"),
            "model_name": model_name,
            "numeric_features": MODEL_FEATURE_COLUMNS,
            "family_column": FAMILY_COLUMN,
            "class_labels": CLASS_LABELS,
            "swap_symmetric_prediction": config.swap_augmentation,
            "numeric_medians": preprocessing["numeric_medians"],
        },
        model_dir / "model.joblib",
    )
    _save_feature_importance(model, preprocessing, model_dir)
    cv_score = float(study.best_value)
    return {
        "model": model_name,
        "status": "ok",
        "tuning_weighted_macro_f1": cv_score,
        "tuning_doi_macro_f1": tuning_metrics["mean_metrics"]["doi_macro_f1"],
        "validation_weighted_macro_f1": validation_metrics["weighted_macro_f1"],
        "validation_macro_f1": validation_metrics["macro_f1"],
        "validation_doi_macro_f1": validation_metrics["doi_macro_f1"],
        "validation_balanced_accuracy": validation_metrics["balanced_accuracy"],
        "validation_accuracy": validation_metrics["accuracy"],
        "validation_weighted_log_loss": validation_metrics["weighted_log_loss"],
        "validation_weighted_ordinal_mae": validation_metrics[
            "weighted_ordinal_mae"
        ],
        "validation_weighted_severe_reversal_rate": validation_metrics[
            "weighted_severe_reversal_rate"
        ],
        "best_trial": int(study.best_trial.number),
    }


def _generate_report(run_dir: Path, comparison: pd.DataFrame) -> None:
    """Generate compact classification figures and a Markdown report."""
    figures = run_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    models = comparison["model"].tolist()
    colors = ["#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]

    metrics = ["tuning_weighted_macro_f1", "validation_weighted_macro_f1",
               "validation_balanced_accuracy", "validation_accuracy"]
    labels = ["Grouped-CV weighted Macro-F1", "Held-out validation weighted Macro-F1",
              "Validation balanced accuracy", "Validation accuracy"]
    x = np.arange(len(models)); width = 0.19
    fig, ax = plt.subplots(figsize=(12, 7))
    for i, (metric, label) in enumerate(zip(metrics, labels)):
        ax.bar(x + (i - 1.5) * width, comparison[metric], width, label=label)
    ax.set_xticks(x, models); ax.set_ylim(0, 1); ax.set_ylabel("Score")
    ax.set_title("Trend Classification Model Performance")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=.25)
    fig.tight_layout(); fig.savefig(figures / "model_metric_comparison.png", dpi=220); plt.close(fig)

    fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 4))
    for ax, model in zip(np.atleast_1d(axes), models):
        payload = json.loads((run_dir / model / "validation_metrics.json").read_text())
        matrix = np.asarray(payload["confusion_matrix"])
        image = ax.imshow(matrix, cmap="Blues")
        for i in range(3):
            for j in range(3): ax.text(j, i, str(matrix[i, j]), ha="center", va="center")
        ax.set_title(model); ax.set_xticks(range(3), CLASS_LABELS, rotation=35, ha="right")
        ax.set_yticks(range(3), CLASS_LABELS); ax.set_xlabel("Predicted")
        if ax is np.atleast_1d(axes)[0]: ax.set_ylabel("True")
    fig.suptitle("Validation Confusion Matrices", fontsize=16, fontweight="bold")
    fig.tight_layout(); fig.savefig(figures / "confusion_matrices.png", dpi=220); plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6)); width = .22
    for i, model in enumerate(models):
        payload = json.loads((run_dir / model / "validation_metrics.json").read_text())
        values = [payload["per_class"][label]["f1"] for label in CLASS_LABELS]
        ax.bar(np.arange(3) + (i - 1.5) * width, values, width, label=model, color=colors[i])
    ax.set_xticks(range(3), CLASS_LABELS); ax.set_ylim(0, 1); ax.set_ylabel("F1")
    ax.set_title("Validation F1 by Class"); ax.legend(); ax.grid(axis="y", alpha=.25)
    fig.tight_layout(); fig.savefig(figures / "per_class_f1.png", dpi=220); plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, model in zip(axes.ravel(), models):
        trials = pd.read_csv(run_dir / model / "optuna_trials.csv")
        value_col = "value" if "value" in trials else "Value"
        values = pd.to_numeric(trials[value_col], errors="coerce")
        ax.plot(values, marker="o", ms=3, alpha=.55, label="trial")
        ax.plot(values.cummax(), color="red", lw=2, label="best")
        ax.set_title(model); ax.set_xlabel("Trial"); ax.set_ylabel("Grouped-CV weighted Macro-F1")
        ax.grid(alpha=.2); ax.legend()
    fig.suptitle("Optuna Optimization History", fontsize=16, fontweight="bold")
    fig.tight_layout(); fig.savefig(figures / "optuna_history.png", dpi=220); plt.close(fig)

    best = str(comparison.iloc[0]["model"])
    importance_path = run_dir / best / "feature_importance.csv"
    if importance_path.exists():
        imp = pd.read_csv(importance_path).head(15).sort_values("importance")
        fig, ax = plt.subplots(figsize=(10, 7)); ax.barh(imp["feature"], imp["importance"], color="#2a9d8f")
        ax.set_title(f"{best} Top 15 Feature Importance"); ax.set_xlabel("Importance")
        fig.tight_layout(); fig.savefig(figures / "best_model_feature_importance.png", dpi=220); plt.close(fig)

    lines = ["# Trend classifier report", "", f"Best model by grouped CV: **{best}**", "",
             comparison.to_markdown(index=False), "", "## Figures", "",
             "- `figures/model_metric_comparison.png`", "- `figures/confusion_matrices.png`",
             "- `figures/per_class_f1.png`", "- `figures/optuna_history.png`",
             "- `figures/best_model_feature_importance.png` (when available)"]
    (run_dir / "model_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def train_classifiers(config: ClassifierConfig | None = None) -> Path:
    """Tune, fit, and compare all requested classifiers."""
    config = config or ClassifierConfig()
    if not config.swap_augmentation:
        raise ValueError(
            "Trend training requires swap augmentation to preserve A/B "
            "direction symmetry."
        )
    _validate_dependencies(config.models)
    if config.n_trials < 1:
        raise ValueError("n_trials must be positive.")
    if config.cv_splits < 2:
        raise ValueError("cv_splits must be at least two.")
    train, validation = _load_data(config)
    run_name = config.run_name or (
        f"trend_cls_v2_f{len(MODEL_FEATURE_COLUMNS)}_family_dsigma1e-4_swap_"
        f"groupcv{config.cv_splits}_optuna{config.n_trials}_seed{config.seed}"
    )
    run_dir = config.output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    data_dir = run_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config.train_path, data_dir / "train.csv")
    shutil.copy2(config.validation_path, data_dir / "validation.csv")

    _write_json(run_dir / "config.json", asdict(config))
    _write_json(
        run_dir / "manifest.json",
        {
            "schema_version": "trend_classifier_run_v1",
            "data_version": "data-trend-v2",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "train_input": portable_path(config.train_path),
            "validation_input": portable_path(config.validation_path),
            "train_sha256": _sha256(config.train_path),
            "validation_sha256": _sha256(config.validation_path),
            "train_rows_original": len(train),
            "train_rows_after_swap": (
                len(train) * 2 if config.swap_augmentation else len(train)
            ),
            "validation_rows": len(validation),
            "train_dois": int(train[GROUP_COLUMN].nunique()),
            "validation_dois": int(validation[GROUP_COLUMN].nunique()),
            "models": config.models,
            "numeric_model_features": MODEL_FEATURE_COLUMNS,
            "categorical_model_features": [FAMILY_COLUMN],
            "model_input_count_before_encoding": len(MODEL_FEATURE_COLUMNS) + 1,
            "target": TARGET_COLUMN,
            "classes": CLASS_LABELS,
            "label_policy": {
                "type": "absolute_conductivity_change",
                "threshold_S_cm-1": TREND_ABSOLUTE_THRESHOLD_S_CM,
                "boundary": "unchanged when abs(delta) <= threshold",
            },
            "tuning_split": (
                "StratifiedGroupKFold on training data grouped by DOI; "
                "held-out validation evaluated only after parameter selection"
            ),
            "cv_splits": config.cv_splits,
            "cv_group_column": GROUP_COLUMN,
            "optuna_objective": "DOI-grouped CV pair-weighted Macro-F1",
            "sample_weight": WEIGHT_COLUMN,
            "swap_augmentation": {
                "enabled": config.swap_augmentation,
                "forward_weight_multiplier": 0.5 if config.swap_augmentation else 1.0,
                "reverse_weight_multiplier": 0.5 if config.swap_augmentation else 0.0,
                "signed_delta_operation": "multiply by -1",
                "mean_and_magnitude_operation": "unchanged",
                "label_operation": "increase/decrease swap; unchanged unchanged",
                "validation_prediction": "average forward and reverse-aligned probabilities",
            },
            "dependency_versions": _dependency_versions(),
        },
    )

    results = []
    total_start = time.monotonic()
    for model_index, model_name in enumerate(config.models, start=1):
        print(
            f"Starting {model_name} ({model_index}/{len(config.models)}); "
            f"target_trials={config.n_trials}",
            flush=True,
        )
        model_start = time.monotonic()
        result = _train_model(
            model_name, train, validation, config, run_dir
        )
        result["elapsed_minutes"] = (time.monotonic() - model_start) / 60.0
        results.append(result)
        pd.DataFrame(results).to_csv(run_dir / "model_comparison.csv", index=False)
        elapsed = time.monotonic() - total_start
        mean_model_time = elapsed / model_index
        eta = mean_model_time * (len(config.models) - model_index)
        print(
            f"Completed {model_name}; elapsed={elapsed / 60:.1f}min; "
            f"remaining_models_eta={eta / 60:.1f}min",
            flush=True,
        )

    comparison = pd.DataFrame(results).sort_values(
        "tuning_weighted_macro_f1", ascending=False
    )
    comparison.to_csv(run_dir / "model_comparison.csv", index=False)
    best_model = str(comparison.iloc[0]["model"])
    (run_dir / "best_model.txt").write_text(best_model + "\n", encoding="utf-8")
    _write_json(
        run_dir / "summary.json",
        {
            "best_model_by_cv": best_model,
            "selection_rule": "highest DOI-grouped CV weighted Macro-F1",
            "held_out_validation_not_used_for_selection": True,
            "comparison": comparison.to_dict(orient="records"),
        },
    )
    _generate_report(run_dir, comparison)
    print(f"Run complete: {run_dir}", flush=True)
    return run_dir


def main() -> None:
    """Train trend classifiers with the default configuration (point-run entry).

    Run directly via ``python main/trend/train_classifier.py`` (the "Run" button).
    Inputs  : data/trend/data-trend-v2-pairs-feature-train.csv
              data/trend/data-trend-v2-pairs-feature-validation.csv
    Output  : runs/trend/<run>/   (per-model model files, metrics.json,
              config.json, validation predictions)

    Note: the default configuration tunes ``n_trials=50`` Optuna trials per
    model and trains all four models; this step can take a while.
    """
    config = ClassifierConfig()
    output_dir = train_classifiers(config)
    print(f"Output dir: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
