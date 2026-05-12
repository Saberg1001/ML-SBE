from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

try:
    import optuna
except ImportError as exc:
    raise SystemExit(
        "Optuna is not installed. Install project dependencies first, for example: "
        "pip install -r requirements.txt"
    ) from exc

from feature_engineering import DEFAULT_TEST, DEFAULT_TRAIN, feature_set_name, prepare_data
from train_models import (
    metrics,
    plot_training_summary,
    save_feature_importance,
    save_json,
    save_predictions,
)


OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs_optuna"


def output_dir(args: argparse.Namespace, model_name: str) -> Path:
    return args.output_root / feature_set_name(args.train) / model_name


def fit_with_optional_weight(model, X_train, y_train, w_train, *, use_weight: bool):
    if use_weight:
        try:
            model.fit(X_train, y_train, sample_weight=w_train)
        except TypeError:
            model.fit(X_train, y_train)
    else:
        model.fit(X_train, y_train)


def fit_lightgbm(
    model,
    X_train,
    y_train,
    w_train,
    X_valid=None,
    y_valid=None,
    *,
    use_weight: bool,
    early_stopping_rounds: int | None,
):
    fit_kwargs = {}
    if use_weight:
        fit_kwargs["sample_weight"] = w_train
    if X_valid is not None and y_valid is not None and early_stopping_rounds:
        fit_kwargs["eval_set"] = [(X_valid, y_valid)]
        fit_kwargs["eval_metric"] = "mae"
        fit_kwargs["callbacks"] = [
            lgb.early_stopping(early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=0),
        ]
    try:
        model.fit(X_train, y_train, **fit_kwargs)
    except TypeError:
        fit_kwargs.pop("sample_weight", None)
        model.fit(X_train, y_train, **fit_kwargs)


def scale_split(X_train, X_valid, columns):
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=columns,
        index=X_train.index,
    )
    X_valid_scaled = pd.DataFrame(
        scaler.transform(X_valid),
        columns=columns,
        index=X_valid.index,
    )
    return X_train_scaled, X_valid_scaled


def trial_result(trial, fold_maes, params, extra: dict | None = None) -> dict:
    result = {
        "trial": trial.number,
        "params": params,
        "mae_mean": float(np.mean(fold_maes)),
        "mae_std": float(np.std(fold_maes)),
        "fold_maes": [float(value) for value in fold_maes],
    }
    if extra:
        result.update(extra)
    return result


def cv_objective(
    trial,
    model_factory,
    param_sampler,
    X,
    y,
    w,
    args,
    *,
    model_name: str,
    use_weight: bool = True,
    scale: bool = False,
    lightgbm_early_stopping: bool = False,
):
    kf = KFold(n_splits=args.cv_splits, shuffle=True, random_state=args.cv_seed)
    params = param_sampler(trial)
    fold_maes = []
    best_iterations = []

    for fold_index, (train_idx, valid_idx) in enumerate(kf.split(X), start=1):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
        w_train = w.iloc[train_idx] if use_weight else None

        if scale:
            X_train, X_valid = scale_split(X_train, X_valid, X.columns)

        model = model_factory(params)
        if lightgbm_early_stopping:
            fit_lightgbm(
                model,
                X_train,
                y_train,
                w_train,
                X_valid,
                y_valid,
                use_weight=use_weight,
                early_stopping_rounds=args.early_stopping_rounds,
            )
            best_iteration = getattr(model, "best_iteration_", None)
            if best_iteration:
                best_iterations.append(int(best_iteration))
        else:
            fit_with_optional_weight(model, X_train, y_train, w_train, use_weight=use_weight)

        pred = model.predict(X_valid)
        fold_maes.append(mean_absolute_error(y_valid, pred))
        trial.report(float(np.mean(fold_maes)), step=fold_index)
        if trial.should_prune():
            raise optuna.TrialPruned()

    extra = {}
    if best_iterations:
        extra["best_iterations"] = best_iterations
        extra["best_n_estimators"] = int(np.median(best_iterations))
        trial.set_user_attr("best_iterations", best_iterations)
        trial.set_user_attr("best_n_estimators", extra["best_n_estimators"])

    result = trial_result(trial, fold_maes, params, extra)
    trial.set_user_attr("result", result)
    return result["mae_mean"]


def make_study(args: argparse.Namespace, model_name: str):
    sampler = optuna.samplers.TPESampler(seed=args.optuna_seed)
    if args.pruner == "median":
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
    else:
        pruner = optuna.pruners.NopPruner()
    study_prefix = args.study_name or feature_set_name(args.train)
    study_name = f"{study_prefix}_{model_name}"
    return optuna.create_study(
        direction="minimize",
        study_name=study_name,
        storage=args.storage,
        load_if_exists=bool(args.storage),
        sampler=sampler,
        pruner=pruner,
    )


def progress_callback(start_time: float, args: argparse.Namespace):
    def callback(study, trial):
        completed = len([item for item in study.trials if item.state == optuna.trial.TrialState.COMPLETE])
        pruned = len([item for item in study.trials if item.state == optuna.trial.TrialState.PRUNED])
        elapsed = time.time() - start_time
        if completed > 0:
            seconds_per_complete = elapsed / completed
            remaining = max(args.n_trials - completed - pruned, 0)
            eta = remaining * seconds_per_complete
            eta_text = f", ETA about {eta / 60:.1f} min"
        else:
            eta_text = ""
        print(
            f"  Trial {trial.number} {trial.state.name}: "
            f"best CV MAE={study.best_value:.4f}, "
            f"completed={completed}, pruned={pruned}, elapsed={elapsed / 60:.1f} min{eta_text}"
        )

    return callback


def clean_trials(study) -> list[dict]:
    results = []
    for trial in study.trials:
        result = trial.user_attrs.get("result")
        if result is not None:
            result = dict(result)
            result["state"] = trial.state.name
            results.append(result)
        else:
            results.append(
                {
                    "trial": trial.number,
                    "params": trial.params,
                    "state": trial.state.name,
                    "mae_mean": trial.value,
                }
            )
    return results


def trial_history_frame(cv_results: list[dict]) -> pd.DataFrame:
    rows = []
    for result in cv_results:
        row = {
            "trial": result.get("trial"),
            "state": result.get("state"),
            "mae_mean": result.get("mae_mean"),
            "mae_std": result.get("mae_std"),
        }
        for key, value in result.get("params", {}).items():
            row[f"param_{key}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def save_optimization_history_plot(frame: pd.DataFrame, path: Path) -> None:
    complete = frame[frame["state"] == "COMPLETE"].dropna(subset=["mae_mean"]).copy()
    if complete.empty:
        return

    complete = complete.sort_values("trial")
    complete["best_so_far"] = complete["mae_mean"].cummin()
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.scatter(complete["trial"], complete["mae_mean"], s=34, alpha=0.72, label="Trial CV MAE")
    ax.plot(complete["trial"], complete["best_so_far"], color="#dc2626", linewidth=2.0, label="Best so far")
    ax.set_xlabel("Trial")
    ax.set_ylabel("5-fold CV MAE")
    ax.set_title("Optuna Optimization History", weight="bold")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_param_importance_outputs(study, model_dir: Path) -> None:
    try:
        importances = optuna.importance.get_param_importances(study)
    except Exception as exc:
        print(f"Could not compute Optuna parameter importances: {exc}")
        return

    if not importances:
        return

    importance_frame = pd.DataFrame(
        [{"parameter": key, "importance": value} for key, value in importances.items()]
    )
    importance_frame.to_csv(model_dir / "optuna_param_importance.csv", index=False)

    plot_frame = importance_frame.sort_values("importance", ascending=True)
    fig, ax = plt.subplots(figsize=(8.0, max(3.8, 0.38 * len(plot_frame))))
    ax.barh(plot_frame["parameter"], plot_frame["importance"], color="#0f766e", alpha=0.9)
    ax.set_xlabel("Importance")
    ax.set_title("Optuna Parameter Importance", weight="bold")
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(model_dir / "optuna_param_importance.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_optimization_outputs(study, cv_results: list[dict], model_dir: Path) -> None:
    frame = trial_history_frame(cv_results)
    if frame.empty:
        return

    frame.to_csv(model_dir / "optuna_trials.csv", index=False)
    save_optimization_history_plot(frame, model_dir / "optuna_optimization_history.png")
    save_param_importance_outputs(study, model_dir)


def fit_final(model, X_train, y_train, w_train, X_test, *, use_weight=True, scale=False):
    scaler = None
    X_train_fit = X_train
    X_test_fit = X_test
    if scale:
        scaler = StandardScaler()
        X_train_fit = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index,
        )
        X_test_fit = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index,
        )

    fit_with_optional_weight(model, X_train_fit, y_train, w_train, use_weight=use_weight)
    train_pred = model.predict(X_train_fit)
    test_pred = model.predict(X_test_fit)
    return model, scaler, train_pred, test_pred


def train_from_optuna(
    model_name,
    model_factory,
    param_sampler,
    data,
    args,
    *,
    use_weight=True,
    scale=False,
    lightgbm_early_stopping=False,
) -> dict:
    print("\n" + "=" * 60)
    print(f"{model_name.upper()} OPTUNA")
    print("=" * 60)
    print(
        f"Search: {args.n_trials} trials, {args.cv_splits}-fold CV, "
        f"sampler=TPE, pruner={args.pruner}"
    )
    start = time.time()
    study = make_study(args, model_name)
    study.optimize(
        lambda trial: cv_objective(
            trial,
            model_factory,
            param_sampler,
            data["X_train"],
            data["y_train"],
            data["w_train"],
            args,
            model_name=model_name,
            use_weight=use_weight,
            scale=scale,
            lightgbm_early_stopping=lightgbm_early_stopping,
        ),
        n_trials=args.n_trials,
        timeout=args.timeout,
        callbacks=[progress_callback(start, args)],
        show_progress_bar=False,
    )

    fixed_trial = optuna.trial.FixedTrial(study.best_trial.params)
    best_params = param_sampler(fixed_trial)
    best_iterations = study.best_trial.user_attrs.get("best_iterations", [])
    best_n_estimators = study.best_trial.user_attrs.get("best_n_estimators")
    if lightgbm_early_stopping and best_n_estimators:
        best_params = dict(best_params)
        best_params["n_estimators"] = int(best_n_estimators)

    model, scaler, train_pred, test_pred = fit_final(
        model_factory(best_params),
        data["X_train"],
        data["y_train"],
        data["w_train"],
        data["X_test"],
        use_weight=use_weight,
        scale=scale,
    )

    train_metrics = metrics(data["y_train"], train_pred)
    test_metrics = metrics(data["y_test"], test_pred)
    elapsed = time.time() - start
    model_dir = output_dir(args, model_name)
    model_dir.mkdir(parents=True, exist_ok=True)

    save_predictions(model_dir / "train_predictions.csv", data["train_ids"], data["y_train"], train_pred)
    save_predictions(model_dir / "test_predictions.csv", data["test_ids"], data["y_test"], test_pred)
    joblib.dump({"model": model, "scaler": scaler, "feature_cols": data["feature_cols"]}, model_dir / "model.joblib")

    if hasattr(model, "feature_importances_"):
        save_feature_importance(model_dir / "feature_importance.csv", data["feature_cols"], model.feature_importances_)

    cv_results = clean_trials(study)
    save_optimization_outputs(study, cv_results, model_dir)
    best_result = study.best_trial.user_attrs.get("result", {})
    result = {
        "model": model_name,
        "feature_set": feature_set_name(args.train),
        "train_path": str(args.train),
        "test_path": str(args.test),
        "n_features": len(data["feature_cols"]),
        "optimization": {
            "method": "optuna_tpe",
            "n_trials": args.n_trials,
            "timeout": args.timeout,
            "cv_splits": args.cv_splits,
            "cv_seed": args.cv_seed,
            "optuna_seed": args.optuna_seed,
            "pruner": args.pruner,
            "lightgbm_early_stopping_rounds": args.early_stopping_rounds
            if lightgbm_early_stopping
            else None,
        },
        "best_params": best_params,
        "cv_best_mae": float(study.best_value),
        "cv_best_mae_std": best_result.get("mae_std"),
        "cv_best_fold_maes": best_result.get("fold_maes"),
        "best_iterations": best_iterations,
        "cv_results": cv_results,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "elapsed_seconds": elapsed,
        "upper_bound_replacement": args.upper_bound_replacement,
        "upper_bound_weight": args.upper_bound_weight,
    }
    save_json(model_dir / "final_results.json", result)

    print(f"Best params: {best_params}")
    if best_iterations:
        print(f"LightGBM CV best iterations: {best_iterations}; final n_estimators={best_params['n_estimators']}")
    print(f"Train: MAE={train_metrics['mae']:.4f}, RMSE={train_metrics['rmse']:.4f}, R2={train_metrics['r2']:.4f}")
    print(f"Test:  MAE={test_metrics['mae']:.4f}, RMSE={test_metrics['rmse']:.4f}, R2={test_metrics['r2']:.4f}")
    print(f"Outputs: {model_dir}")
    return result


def sample_lightgbm_params(trial):
    max_depth = trial.suggest_categorical("max_depth", [-1, 3, 4, 5, 6, 7, 8, 10])
    max_leaves = 63 if max_depth == -1 else min(63, 2**max_depth)
    return {
        "n_estimators": trial.suggest_int("n_estimators", 500, 3000, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 7, max_leaves),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 60),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 50.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 20.0, log=True),
        "max_depth": max_depth,
    }


def train_lightgbm(data: dict, args: argparse.Namespace) -> dict:
    def factory(params):
        return lgb.LGBMRegressor(
            objective="regression",
            verbosity=-1,
            n_jobs=-1,
            random_state=42,
            subsample_freq=1,
            **params,
        )

    return train_from_optuna(
        "lightgbm",
        factory,
        sample_lightgbm_params,
        data,
        args,
        use_weight=True,
        lightgbm_early_stopping=not args.disable_lightgbm_early_stopping,
    )


def sample_random_forest_params(trial):
    max_features_type = trial.suggest_categorical("max_features_type", ["sqrt", "log2", "float"])
    if max_features_type == "float":
        max_features = trial.suggest_float("max_features_float", 0.4, 1.0)
    else:
        max_features = max_features_type
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1200, step=100),
        "max_depth": trial.suggest_categorical("max_depth", [None, 5, 7, 10, 15, 20, 30]),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": max_features,
    }


def train_random_forest(data: dict, args: argparse.Namespace) -> dict:
    return train_from_optuna(
        "random_forest",
        lambda params: RandomForestRegressor(random_state=42, n_jobs=-1, **params),
        sample_random_forest_params,
        data,
        args,
        use_weight=True,
    )


def sample_decision_tree_params(trial):
    return {
        "max_depth": trial.suggest_categorical("max_depth", [None, 3, 5, 7, 10, 15, 20]),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 30),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 15),
        "max_features": trial.suggest_categorical("max_features", [None, "sqrt", "log2"]),
    }


def train_decision_tree(data: dict, args: argparse.Namespace) -> dict:
    return train_from_optuna(
        "decision_tree",
        lambda params: DecisionTreeRegressor(random_state=42, **params),
        sample_decision_tree_params,
        data,
        args,
        use_weight=True,
    )


def sample_mlp_params(trial):
    layer_name = trial.suggest_categorical("hidden_layer_sizes", ["64", "64_32", "128_64", "128_64_32"])
    layer_map = {
        "64": (64,),
        "64_32": (64, 32),
        "128_64": (128, 64),
        "128_64_32": (128, 64, 32),
    }
    return {
        "hidden_layer_sizes": layer_map[layer_name],
        "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
        "alpha": trial.suggest_float("alpha", 1e-5, 1.0, log=True),
        "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True),
        "max_iter": trial.suggest_categorical("max_iter", [500, 1000, 1500, 2000]),
    }


def train_mlp(data: dict, args: argparse.Namespace) -> dict:
    return train_from_optuna(
        "mlp",
        lambda params: MLPRegressor(
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            **params,
        ),
        sample_mlp_params,
        data,
        args,
        use_weight=False,
        scale=True,
    )


def sample_ngboost_params(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1500, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "minibatch_frac": trial.suggest_float("minibatch_frac", 0.5, 1.0),
    }


def train_ngboost(data: dict, args: argparse.Namespace) -> dict | None:
    try:
        from ngboost import NGBRegressor
        from ngboost.distns import Normal
    except ImportError:
        print("NGBoost is not installed; skipping.")
        return None

    return train_from_optuna(
        "ngboost",
        lambda params: NGBRegressor(Dist=Normal, random_state=42, verbose=False, **params),
        sample_ngboost_params,
        data,
        args,
        use_weight=False,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train models with Optuna hyperparameter optimization.")
    parser.add_argument("--model", choices=["lightgbm", "rf", "dt", "mlp", "ngboost", "all"], default="all")
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--no-interactions", action="store_true")
    parser.add_argument("--remove-upper-bound", action="store_true")
    parser.add_argument("--upper-bound-threshold", type=float, default=1e-10)
    parser.add_argument("--upper-bound-replacement", type=float, default=1e-11)
    parser.add_argument("--upper-bound-weight", type=float, default=0.3)
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--timeout", type=int, default=None, help="Maximum seconds per selected model.")
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--cv-seed", type=int, default=42)
    parser.add_argument("--optuna-seed", type=int, default=42)
    parser.add_argument("--study-name", default=None)
    parser.add_argument("--storage", default=None, help="Optional Optuna storage URL, for example sqlite:///study.db.")
    parser.add_argument("--pruner", choices=["none", "median"], default="median")
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    parser.add_argument("--disable-lightgbm-early-stopping", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    feature_name = feature_set_name(args.train)
    data = prepare_data(
        train_path=args.train,
        test_path=args.test,
        output_dir=args.output_root / feature_name / "data",
        remove_upper_bound=args.remove_upper_bound,
        add_interactions=not args.no_interactions,
        upper_bound_threshold=args.upper_bound_threshold,
        upper_bound_replacement=args.upper_bound_replacement,
        upper_bound_weight=args.upper_bound_weight,
    )

    trainers = {
        "lightgbm": train_lightgbm,
        "rf": train_random_forest,
        "dt": train_decision_tree,
        "mlp": train_mlp,
        "ngboost": train_ngboost,
    }
    selected = list(trainers) if args.model == "all" else [args.model]
    results = []
    for name in selected:
        result = trainers[name](data, args)
        if result is not None:
            results.append(result)

    comparison = [
        {
            "model": item["model"],
            "feature_set": item["feature_set"],
            "cv_mae": item["cv_best_mae"],
            "train_mae": item["train_metrics"]["mae"],
            "train_rmse": item["train_metrics"]["rmse"],
            "train_r2": item["train_metrics"]["r2"],
            "test_mae": item["test_metrics"]["mae"],
            "test_rmse": item["test_metrics"]["rmse"],
            "test_r2": item["test_metrics"]["r2"],
        }
        for item in results
    ]
    comparison_dir = args.output_root / feature_name
    pd.DataFrame(comparison).to_csv(comparison_dir / "model_comparison.csv", index=False)
    save_json(comparison_dir / "model_comparison.json", {"results": results})
    plot_training_summary(results, comparison, comparison_dir)
    print(f"\nComparison saved to {comparison_dir}")


if __name__ == "__main__":
    main()
