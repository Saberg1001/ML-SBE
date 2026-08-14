from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

try:
    import optuna
except ImportError as exc:
    raise RuntimeError("Optuna is required for main.train") from exc

from .features import FAMILY_COLUMN, family_code_mapping, infer_feature_columns
from .paths import RUNS_DIR


DEFAULT_OUTPUT_ROOT = RUNS_DIR / "absolute"
TARGET_COLUMN = "log10_conductivity"
MODEL_NAMES = ("lightgbm", "random_forest", "decision_tree", "mlp", "ngboost")


@dataclass
class TrainConfig:
    """Options for model training, Optuna tuning, and output packaging.

    model_name:
        Model to train. Supported values are "all", "lightgbm",
        "random_forest", "decision_tree", "mlp", and "ngboost". The default
        "all" trains every available model and selects the lowest test MAE.
    n_trials:
        Number of Optuna trials per selected model.
    cv_splits:
        Number of KFold splits used inside each Optuna trial.
    seed:
        Random seed for cross-validation and deterministic estimators.
    optuna_seed:
        Random seed for Optuna's sampler.
    output_root:
        Root directory for saved runs.
    run_name:
        Optional exact output directory name. If None, a name is built from
        dataset_name, feature count, model_name, n_trials, and seed.
    dataset_name:
        Text tag used in auto-generated run names.
    feature_columns:
        Optional explicit model feature list. If None, numeric feature columns
        are inferred from the train dataframe.
    target_column:
        Regression target column; default is log10_conductivity.
    verbose:
        If True, print Optuna progress with elapsed time and ETA.
    """

    model_name: str = "all"
    n_trials: int = 50
    cv_splits: int = 5
    seed: int = 42
    optuna_seed: int = 42
    output_root: Path = DEFAULT_OUTPUT_ROOT
    run_name: str | None = None
    dataset_name: str = "ionic_main_random_filter_gt1e-6_family"
    feature_columns: list[str] | None = None
    target_column: str = TARGET_COLUMN
    verbose: bool = True


@dataclass
class TrainResult:
    output_dir: Path
    comparison: pd.DataFrame
    best_model: str
    results: dict
    feature_columns: list[str]


def _json_ready(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
    }


def _load_frame(data: pd.DataFrame | str | Path) -> tuple[pd.DataFrame, Path | None]:
    if isinstance(data, pd.DataFrame):
        return data.copy(), None
    path = Path(data)
    return pd.read_csv(path), path


def _feature_columns(train: pd.DataFrame, config: TrainConfig) -> list[str]:
    if config.feature_columns is not None:
        return list(config.feature_columns)
    return infer_feature_columns(train)


def _prepare_matrix(train: pd.DataFrame, test: pd.DataFrame, feature_columns: list[str], target_column: str):
    X_train = train.reindex(columns=feature_columns).apply(pd.to_numeric, errors="coerce")
    X_test = test.reindex(columns=feature_columns).apply(pd.to_numeric, errors="coerce")
    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.replace([np.inf, -np.inf], np.nan)
    medians = X_train.median().fillna(0.0)
    X_train = X_train.fillna(medians)
    X_test = X_test.fillna(medians)
    y_train = pd.to_numeric(train[target_column], errors="coerce")
    y_test = pd.to_numeric(test[target_column], errors="coerce")
    weights = pd.to_numeric(train.get("sample_weight", pd.Series(1.0, index=train.index)), errors="coerce").fillna(1.0)
    return X_train, X_test, y_train, y_test, weights, medians


def _fit_optional_weight(model, X, y, weights, use_weight: bool):
    if use_weight:
        try:
            model.fit(X, y, sample_weight=weights)
            return
        except TypeError:
            pass
    model.fit(X, y)


def _scale_fit(X_train: pd.DataFrame, X_test: pd.DataFrame):
    scaler = StandardScaler()
    train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    return train_scaled, test_scaled, scaler


def _sample_lightgbm(trial):
    max_depth = trial.suggest_categorical("max_depth", [-1, 3, 4, 5, 6, 7, 8, 10])
    max_leaves = 63 if max_depth == -1 else min(63, 2**max_depth)
    return {
        "n_estimators": trial.suggest_int("n_estimators", 300, 2500, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 7, max_leaves),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 60),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 50.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 20.0, log=True),
        "max_depth": max_depth,
    }


def _sample_random_forest(trial):
    max_features_type = trial.suggest_categorical("max_features_type", ["sqrt", "log2", "float"])
    max_features = trial.suggest_float("max_features_float", 0.4, 1.0) if max_features_type == "float" else max_features_type
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000, step=100),
        "max_depth": trial.suggest_categorical("max_depth", [None, 5, 7, 10, 15, 20, 30]),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": max_features,
    }


def _sample_decision_tree(trial):
    return {
        "max_depth": trial.suggest_categorical("max_depth", [None, 3, 5, 7, 10, 15, 20]),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 30),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 15),
        "max_features": trial.suggest_categorical("max_features", [None, "sqrt", "log2"]),
    }


def _sample_mlp(trial):
    layers = {
        "64": (64,),
        "64_32": (64, 32),
        "128_64": (128, 64),
        "128_64_32": (128, 64, 32),
    }
    layer_name = trial.suggest_categorical("hidden_layer_sizes", list(layers))
    return {
        "hidden_layer_sizes": layers[layer_name],
        "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
        "alpha": trial.suggest_float("alpha", 1e-5, 1.0, log=True),
        "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True),
        "max_iter": trial.suggest_categorical("max_iter", [500, 1000, 1500]),
    }


def _sample_ngboost(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1200, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "minibatch_frac": trial.suggest_float("minibatch_frac", 0.5, 1.0),
    }


def _model_spec(model_name: str):
    if model_name == "lightgbm":
        return (
            lambda params: lgb.LGBMRegressor(
                objective="regression",
                verbosity=-1,
                n_jobs=-1,
                random_state=42,
                subsample_freq=1,
                **params,
            ),
            _sample_lightgbm,
            True,
            False,
        )
    if model_name == "random_forest":
        return (
            lambda params: RandomForestRegressor(random_state=42, n_jobs=-1, **params),
            _sample_random_forest,
            True,
            False,
        )
    if model_name == "decision_tree":
        return (
            lambda params: DecisionTreeRegressor(random_state=42, **params),
            _sample_decision_tree,
            True,
            False,
        )
    if model_name == "mlp":
        return (
            lambda params: MLPRegressor(random_state=42, early_stopping=True, validation_fraction=0.1, **params),
            _sample_mlp,
            False,
            True,
        )
    if model_name == "ngboost":
        try:
            from ngboost import NGBRegressor
            from ngboost.distns import Normal
        except ImportError as exc:
            raise RuntimeError("NGBoost is not installed") from exc
        return (
            lambda params: NGBRegressor(Dist=Normal, random_state=42, verbose=False, **params),
            _sample_ngboost,
            False,
            False,
        )
    raise ValueError(f"Unsupported model_name: {model_name}")


def _optimize_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    weights: pd.Series,
    config: TrainConfig,
):
    factory, sampler, use_weight, scale = _model_spec(model_name)
    kfold = KFold(n_splits=config.cv_splits, shuffle=True, random_state=config.seed)
    start = time.time()

    def objective(trial):
        params = sampler(trial)
        fold_maes = []
        for fold_index, (train_idx, valid_idx) in enumerate(kfold.split(X_train), start=1):
            fold_X_train = X_train.iloc[train_idx]
            fold_X_valid = X_train.iloc[valid_idx]
            fold_y_train = y_train.iloc[train_idx]
            fold_y_valid = y_train.iloc[valid_idx]
            fold_weights = weights.iloc[train_idx]
            if scale:
                fold_X_train, fold_X_valid, _ = _scale_fit(fold_X_train, fold_X_valid)
            model = factory(params)
            _fit_optional_weight(model, fold_X_train, fold_y_train, fold_weights, use_weight)
            pred = model.predict(fold_X_valid)
            fold_maes.append(mean_absolute_error(fold_y_valid, pred))
            trial.report(float(np.mean(fold_maes)), step=fold_index)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr("fold_maes", [float(value) for value in fold_maes])
        return float(np.mean(fold_maes))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=config.optuna_seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
    )

    def callback(study_obj, trial):
        if not config.verbose:
            return
        completed = len([item for item in study_obj.trials if item.state == optuna.trial.TrialState.COMPLETE])
        elapsed = time.time() - start
        if completed:
            eta = max(config.n_trials - len(study_obj.trials), 0) * elapsed / completed
            print(
                f"{model_name} trial {trial.number}: best CV MAE={study_obj.best_value:.4f}; "
                f"elapsed={elapsed / 60:.1f} min; ETA={eta / 60:.1f} min",
                flush=True,
            )

    study.optimize(objective, n_trials=config.n_trials, callbacks=[callback], show_progress_bar=False)
    return study, factory, sampler, use_weight, scale


def _best_sampled_params(study, sampler) -> dict:
    """Rebuild estimator params from Optuna's raw trial params."""
    fixed_trial = optuna.trial.FixedTrial(study.best_trial.params)
    return sampler(fixed_trial)


def _fit_final(factory, params, use_weight, scale, X_train, y_train, weights, X_test):
    scaler = None
    fit_X_train = X_train
    fit_X_test = X_test
    if scale:
        fit_X_train, fit_X_test, scaler = _scale_fit(X_train, X_test)
    model = factory(params)
    _fit_optional_weight(model, fit_X_train, y_train, weights, use_weight)
    return model, scaler, model.predict(fit_X_train), model.predict(fit_X_test)


def _save_predictions(path: Path, ids, y_true, y_pred) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "ID": ids.to_numpy() if hasattr(ids, "to_numpy") else ids,
            "y_true": np.asarray(y_true),
            "y_pred": np.asarray(y_pred),
            "residual": np.asarray(y_true) - np.asarray(y_pred),
        }
    ).to_csv(path, index=False)


def _save_feature_importance(model, feature_columns: list[str], path: Path) -> None:
    if not hasattr(model, "feature_importances_"):
        return
    importance = np.asarray(model.feature_importances_, dtype=float)
    if importance.ndim > 1:
        importance = importance.mean(axis=tuple(range(importance.ndim - 1)))
    importance = importance.reshape(-1)
    if len(importance) != len(feature_columns):
        raise ValueError(
            f"Feature importance length {len(importance)} does not match "
            f"feature count {len(feature_columns)}"
        )
    frame = pd.DataFrame({"feature": feature_columns, "importance": importance})
    frame.sort_values("importance", ascending=False).to_csv(path, index=False)


def _plot_model_diagnostics(model_dir: Path, figures_dir: Path) -> None:
    """Write parity, residual, optimization, and importance figures."""
    import matplotlib.pyplot as plt

    train_predictions = pd.read_csv(model_dir / "train_predictions.csv")
    test_predictions = pd.read_csv(model_dir / "test_predictions.csv")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    values = pd.concat([train_predictions, test_predictions], ignore_index=True)
    lower = float(min(values["y_true"].min(), values["y_pred"].min()))
    upper = float(max(values["y_true"].max(), values["y_pred"].max()))
    for axis, (label, frame, color) in zip(
        axes,
        (("Train", train_predictions, "#1f77b4"), ("Test", test_predictions, "#d62728")),
    ):
        axis.scatter(frame["y_true"], frame["y_pred"], alpha=0.7, s=24, color=color)
        axis.plot([lower, upper], [lower, upper], "k--", linewidth=1)
        score = metrics(frame["y_true"], frame["y_pred"])
        axis.set_title(f"{label}: MAE={score['mae']:.3f}, R²={score['r2']:.3f}")
        axis.set_xlabel("True log10 conductivity")
        axis.set_ylabel("Predicted log10 conductivity")
        axis.grid(alpha=0.2)
    fig.suptitle(f"{model_dir.name} prediction parity")
    fig.savefig(figures_dir / f"{model_dir.name}_prediction_parity.png", dpi=300)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 5), constrained_layout=True)
    for label, frame, color in (
        ("Train", train_predictions, "#1f77b4"),
        ("Test", test_predictions, "#d62728"),
    ):
        axis.scatter(frame["y_pred"], frame["residual"], label=label, alpha=0.7, s=24, color=color)
    axis.axhline(0.0, color="black", linestyle="--", linewidth=1)
    axis.set_xlabel("Predicted log10 conductivity")
    axis.set_ylabel("Residual (true - predicted)")
    axis.set_title(f"{model_dir.name} residual analysis")
    axis.legend()
    axis.grid(alpha=0.2)
    fig.savefig(figures_dir / f"{model_dir.name}_residuals.png", dpi=300)
    plt.close(fig)

    trials = pd.read_csv(model_dir / "optuna_trials.csv")
    completed = trials[pd.to_numeric(trials["value"], errors="coerce").notna()].copy()
    if not completed.empty:
        completed["best_value"] = completed["value"].cummin()
        fig, axis = plt.subplots(figsize=(7, 5), constrained_layout=True)
        axis.scatter(completed["trial"], completed["value"], alpha=0.55, label="Trial CV MAE")
        axis.plot(completed["trial"], completed["best_value"], color="#d62728", label="Best CV MAE")
        axis.set_xlabel("Trial")
        axis.set_ylabel("Cross-validation MAE")
        axis.set_title(f"{model_dir.name} optimization history")
        axis.legend()
        axis.grid(alpha=0.2)
        fig.savefig(figures_dir / f"{model_dir.name}_optimization_history.png", dpi=300)
        plt.close(fig)

    importance_path = model_dir / "feature_importance.csv"
    if importance_path.exists():
        importance = pd.read_csv(importance_path).head(20).sort_values("importance")
        fig, axis = plt.subplots(figsize=(9, 7), constrained_layout=True)
        axis.barh(importance["feature"], importance["importance"], color="#2ca02c")
        axis.set_xlabel("Importance")
        axis.set_title(f"{model_dir.name} top feature importance")
        axis.grid(axis="x", alpha=0.2)
        fig.savefig(figures_dir / f"{model_dir.name}_feature_importance.png", dpi=300)
        plt.close(fig)


def _display_model_name(model_name: str) -> str:
    return {
        "lightgbm": "LightGBM",
        "random_forest": "Random Forest",
        "decision_tree": "Decision Tree",
        "mlp": "MLP",
        "ngboost": "NGBoost",
    }.get(model_name, model_name)


def _saved_prediction_metrics(output_dir: Path, model_names: list[str]) -> pd.DataFrame:
    rows = []
    for model_name in model_names:
        train = pd.read_csv(output_dir / model_name / "train_predictions.csv")
        test = pd.read_csv(output_dir / model_name / "test_predictions.csv")
        train_scores = metrics(train["y_true"], train["y_pred"])
        test_scores = metrics(test["y_true"], test["y_pred"])
        rows.append({
            "model": model_name,
            "Train MAE": train_scores["mae"],
            "Test MAE": test_scores["mae"],
            "Train RMSE": train_scores["rmse"],
            "Test RMSE": test_scores["rmse"],
            "Train R²": train_scores["r2"],
            "Test R²": test_scores["r2"],
        })
    return pd.DataFrame(rows)


def _draw_legacy_parity(axis, output_dir: Path, model_name: str, split: str) -> None:
    frame = pd.read_csv(output_dir / model_name / f"{split}_predictions.csv")
    lower = float(min(frame["y_true"].min(), frame["y_pred"].min()))
    upper = float(max(frame["y_true"].max(), frame["y_pred"].max()))
    padding = max((upper - lower) * 0.04, 0.1)
    axis.scatter(
        frame["y_true"], frame["y_pred"], s=70, alpha=0.75,
        color="cornflowerblue", edgecolor="royalblue", linewidth=0.6,
    )
    axis.plot([lower, upper], [lower, upper], "--", color="#e52421", linewidth=2)
    axis.set_xlim(lower - padding, upper + padding)
    axis.set_ylim(lower - padding, upper + padding)
    axis.set_title(_display_model_name(model_name), fontsize=18, fontweight="bold", pad=12)
    axis.set_xlabel("True log10(conductivity)", fontsize=15)
    axis.set_ylabel("Predicted log10(conductivity)", fontsize=15)
    axis.tick_params(labelsize=12)
    axis.grid(alpha=0.25)


def _plot_train_test_parity(output_dir: Path, figures_dir: Path, model_names: list[str]) -> None:
    import matplotlib.pyplot as plt

    count = len(model_names)
    fig, axes = plt.subplots(2, count, figsize=(6 * count, 11), squeeze=False, constrained_layout=True)
    for column, model_name in enumerate(model_names):
        _draw_legacy_parity(axes[0, column], output_dir, model_name, "train")
        _draw_legacy_parity(axes[1, column], output_dir, model_name, "test")
        axes[0, column].set_title(f"{_display_model_name(model_name)} — Train", fontsize=17, fontweight="bold")
        axes[1, column].set_title(f"{_display_model_name(model_name)} — Test", fontsize=17, fontweight="bold")
    fig.suptitle("Train (top) / Test (bottom)", fontsize=20, fontweight="bold")
    if count == 1:
        output_path = figures_dir / f"{model_names[0]}_prediction_parity.png"
    else:
        output_path = figures_dir / "predicted_vs_true_all_models.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _draw_metrics_table(axis, metric_frame: pd.DataFrame) -> None:
    axis.axis("off")
    headers = ["Model", "Train MAE", "Test MAE", "Train RMSE", "Test RMSE", "Train R²", "Test R²"]
    cells = [
        [_display_model_name(row["model"]), *[f"{row[column]:.3f}" for column in headers[1:]]]
        for _, row in metric_frame.iterrows()
    ]
    table = axis.table(
        cellText=cells,
        colLabels=headers,
        cellLoc="center",
        colWidths=[0.20, *([0.133] * 6)],
        loc="center",
        bbox=[0, 0, 1, 0.82],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#d5dbe3")
        if row == 0:
            cell.set_facecolor("#202b3b")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_text_props(fontweight="bold")
    axis.set_title("Train/Test Metrics Across Models", fontsize=20, fontweight="bold", pad=10)


def _draw_top10_importance(axis, output_dir: Path, model_name: str) -> None:
    if not (output_dir / model_name / "feature_importance.csv").exists():
        model_name = None
    if model_name is None:
        axis.axis("off")
        axis.text(0.5, 0.5, "Feature importance unavailable", ha="center", va="center")
        return
    importance = pd.read_csv(output_dir / model_name / "feature_importance.csv").head(10).copy()
    total = float(pd.read_csv(output_dir / model_name / "feature_importance.csv")["importance"].sum())
    importance["share"] = importance["importance"] / total * 100 if total else 0.0
    importance = importance.sort_values("share")
    axis.barh(importance["feature"], importance["share"], color="#2a8882")
    for index, value in enumerate(importance["share"]):
        axis.text(value + 0.12, index, f"{value:.1f}%", va="center", fontsize=10)
    axis.set_xlabel(f"Share of total {_display_model_name(model_name)} importance (%)")
    axis.set_title(f"{_display_model_name(model_name)} Top 10 Feature Importance Share", fontweight="bold")
    axis.grid(axis="x", alpha=0.2)


def _plot_legacy_summary(
    output_dir: Path,
    figures_dir: Path,
    model_names: list[str],
    metric_frame: pd.DataFrame,
    split: str,
) -> None:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(16, 12), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.25, 1.0], width_ratios=[2.0, 1.0])
    parity = fig.add_subplot(grid[0, :])
    _draw_legacy_parity(parity, output_dir, model_names[0], split)
    table_axis = fig.add_subplot(grid[1, 0])
    _draw_metrics_table(table_axis, metric_frame)
    importance_axis = fig.add_subplot(grid[1, 1])
    _draw_top10_importance(importance_axis, output_dir, model_names[0])
    fig.suptitle("Model Performance Summary", fontsize=24, fontweight="bold")
    suffix = "_train" if split == "train" else ""
    fig.savefig(figures_dir / f"model_performance_summary_combined{suffix}.png", dpi=300)
    plt.close(fig)


def _plot_total_summary(
    output_dir: Path,
    figures_dir: Path,
    best_model_name: str,
    metric_frame: pd.DataFrame,
) -> None:
    """Combine train/test parity, metrics, and importance in one figure."""
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(16, 17), constrained_layout=True)
    grid = fig.add_gridspec(3, 2, height_ratios=[1.05, 1.05, 1.0], width_ratios=[2.0, 1.0])

    train_axis = fig.add_subplot(grid[0, :])
    _draw_legacy_parity(train_axis, output_dir, best_model_name, "train")
    train_axis.set_title(f"{_display_model_name(best_model_name)} — Train", fontsize=20, fontweight="bold")

    test_axis = fig.add_subplot(grid[1, :])
    _draw_legacy_parity(test_axis, output_dir, best_model_name, "test")
    test_axis.set_title(f"{_display_model_name(best_model_name)} — Test", fontsize=20, fontweight="bold")

    table_axis = fig.add_subplot(grid[2, 0])
    _draw_metrics_table(table_axis, metric_frame)
    importance_axis = fig.add_subplot(grid[2, 1])
    _draw_top10_importance(importance_axis, output_dir, best_model_name)

    fig.suptitle("Model Performance Overview", fontsize=26, fontweight="bold")
    fig.savefig(figures_dir / "model_performance_overview.png", dpi=300)
    plt.close(fig)

def generate_training_figures(output_dir: str | Path) -> Path:
    """Generate mandatory diagnostic figures from saved training artifacts."""
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    comparison = pd.read_csv(output_dir / "model_comparison.csv")
    successful = comparison[comparison["status"].eq("ok")].copy()
    if successful.empty:
        raise RuntimeError("No successful models are available for figure generation")

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    best_model_name = str(successful.sort_values("test_mae").iloc[0]["model"])
    all_model_names = [str(name) for name in successful["model"]]
    _plot_model_diagnostics(output_dir / best_model_name, figures_dir)

    metric_frame = _saved_prediction_metrics(output_dir, all_model_names)
    _plot_train_test_parity(output_dir, figures_dir, all_model_names)

    fig, axis = plt.subplots(figsize=(14, 5), constrained_layout=True)
    _draw_metrics_table(axis, metric_frame)
    fig.savefig(figures_dir / "train_test_metrics_table.png", dpi=300)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 6), constrained_layout=True)
    _draw_top10_importance(axis, output_dir, best_model_name)
    fig.savefig(figures_dir / f"{best_model_name}_top10_feature_share.png", dpi=300)
    plt.close(fig)

    _plot_legacy_summary(output_dir, figures_dir, [best_model_name], metric_frame, "test")
    _plot_legacy_summary(output_dir, figures_dir, [best_model_name], metric_frame, "train")
    _plot_total_summary(output_dir, figures_dir, best_model_name, metric_frame)

    metrics_columns = ["cv_mae", "train_mae", "test_mae", "test_rmse"]
    available = [column for column in metrics_columns if column in successful.columns]
    figure_data = successful.set_index("model")[available]
    fig, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    figure_data.plot(kind="bar", ax=axis)
    axis.set_ylabel("Score")
    axis.set_title("Model performance summary")
    axis.tick_params(axis="x", rotation=0)
    axis.grid(axis="y", alpha=0.2)
    fig.savefig(figures_dir / "model_performance_summary.png", dpi=300)
    plt.close(fig)
    return figures_dir


def _family_mapping_from_frames(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, int]:
    if FAMILY_COLUMN not in train.columns and FAMILY_COLUMN not in test.columns:
        return {}
    labels = pd.concat(
        [
            train[FAMILY_COLUMN] if FAMILY_COLUMN in train.columns else pd.Series(dtype=object),
            test[FAMILY_COLUMN] if FAMILY_COLUMN in test.columns else pd.Series(dtype=object),
        ],
        ignore_index=True,
    )
    return family_code_mapping(labels)


def _run_name(config: TrainConfig, feature_count: int) -> str:
    if config.run_name:
        return config.run_name
    model_tag = config.model_name if config.model_name != "all" else "all_models"
    return f"{config.dataset_name}_{feature_count}_features_{model_tag}_trials{config.n_trials}_seed{config.seed}"


def train_model(
    train_data: pd.DataFrame | str | Path,
    test_data: pd.DataFrame | str | Path,
    config: TrainConfig | None = None,
) -> TrainResult:
    """Train selected models, evaluate train/test metrics, and save artifacts."""

    config = config or TrainConfig()
    train, train_path = _load_frame(train_data)
    test, test_path = _load_frame(test_data)
    feature_columns = _feature_columns(train, config)
    family_mapping = _family_mapping_from_frames(train, test)
    X_train, X_test, y_train, y_test, weights, medians = _prepare_matrix(
        train,
        test,
        feature_columns,
        config.target_column,
    )

    output_dir = Path(config.output_root) / _run_name(config, len(feature_columns))
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(data_dir / "train.csv", index=False)
    test.to_csv(data_dir / "test.csv", index=False)
    X_train.to_csv(data_dir / "X_train.csv", index=False)
    X_test.to_csv(data_dir / "X_test.csv", index=False)
    y_train.to_csv(data_dir / "y_train.csv", index=False, header=[config.target_column])
    y_test.to_csv(data_dir / "y_test.csv", index=False, header=[config.target_column])
    medians.to_csv(data_dir / "feature_medians.csv", header=["median"])
    (data_dir / "feature_list.txt").write_text("\n".join(feature_columns) + "\n", encoding="utf-8")

    selected_models = list(MODEL_NAMES) if config.model_name == "all" else [config.model_name]
    results = {}
    rows = []
    for model_name in selected_models:
        try:
            study, factory, sampler, use_weight, scale = _optimize_model(model_name, X_train, y_train, weights, config)
        except RuntimeError as exc:
            rows.append({"model": model_name, "status": "skipped", "message": str(exc)})
            continue

        best_params = _best_sampled_params(study, sampler)
        model, scaler, train_pred, test_pred = _fit_final(
            factory,
            best_params,
            use_weight,
            scale,
            X_train,
            y_train,
            weights,
            X_test,
        )
        model_dir = output_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        train_metrics = metrics(y_train, train_pred)
        test_metrics = metrics(y_test, test_pred)
        _save_predictions(model_dir / "train_predictions.csv", train.get("ID", train.index), y_train, train_pred)
        _save_predictions(model_dir / "test_predictions.csv", test.get("ID", test.index), y_test, test_pred)
        _save_feature_importance(model, feature_columns, model_dir / "feature_importance.csv")
        pd.DataFrame(
            [
                {
                    "trial": trial.number,
                    "state": trial.state.name,
                    "value": trial.value,
                    **{f"param_{key}": value for key, value in trial.params.items()},
                }
                for trial in study.trials
            ]
        ).to_csv(model_dir / "optuna_trials.csv", index=False)

        artifact = {
            "model": model,
            "scaler": scaler,
            "feature_cols": feature_columns,
            "feature_medians": medians,
            "family_mapping": family_mapping,
        }
        joblib.dump(artifact, model_dir / "model.joblib")
        result = {
            "model": model_name,
            "status": "ok",
            "best_params": best_params,
            "cv_best_mae": float(study.best_value),
            "cv_best_fold_maes": study.best_trial.user_attrs.get("fold_maes", []),
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
        }
        save_json(model_dir / "final_results.json", result)
        results[model_name] = result
        rows.append(
            {
                "model": model_name,
                "status": "ok",
                "cv_mae": result["cv_best_mae"],
                "train_mae": train_metrics["mae"],
                "test_mae": test_metrics["mae"],
                "test_rmse": test_metrics["rmse"],
                "test_r2": test_metrics["r2"],
            }
        )

    comparison = pd.DataFrame(rows)
    comparison.to_csv(output_dir / "model_comparison.csv", index=False)
    ok_comparison = comparison[comparison["status"].eq("ok")] if not comparison.empty else comparison
    best_model = ""
    if not ok_comparison.empty:
        best_model = str(ok_comparison.sort_values("test_mae").iloc[0]["model"])
        (output_dir / "best_model.txt").write_text(best_model + "\n", encoding="utf-8")
        generate_training_figures(output_dir)

    schema = {
        "feature_columns": feature_columns,
        "target_column": config.target_column,
        "train_path": str(train_path) if train_path else None,
        "test_path": str(test_path) if test_path else None,
        "family_mapping": family_mapping,
    }
    save_json(output_dir / "config.json", asdict(config))
    save_json(output_dir / "feature_schema.json", schema)
    save_json(
        output_dir / "summary.json",
        {
            "best_model": best_model,
            "n_features": len(feature_columns),
            "n_train": len(train),
            "n_test": len(test),
            "family_mapping": family_mapping,
            "results": results,
        },
    )
    return TrainResult(
        output_dir=output_dir,
        comparison=comparison,
        best_model=best_model,
        results=results,
        feature_columns=feature_columns,
    )
