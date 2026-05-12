from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MPL_CACHE_DIR = ROOT / ".matplotlib_cache"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

import joblib
import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from feature_engineering import (
    DEFAULT_TEST,
    DEFAULT_TRAIN,
    feature_set_name,
    prepare_data,
)
from feature_labels import display_feature_label


OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs"
MODEL_LABELS = {
    "lightgbm": "LightGBM",
    "random_forest": "Random Forest",
    "decision_tree": "Decision Tree",
    "mlp": "MLP",
    "ngboost": "NGBoost",
}


def metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, default=str)
        file.write("\n")


def output_dir(args: argparse.Namespace, model_name: str) -> Path:
    return args.output_root / feature_set_name(args.train) / model_name


def save_predictions(path: Path, ids, y_true, y_pred) -> None:
    pd.DataFrame(
        {
            "ID": ids.to_numpy() if hasattr(ids, "to_numpy") else ids,
            "y_true": np.asarray(y_true),
            "y_pred": np.asarray(y_pred),
            "residual": np.asarray(y_true) - np.asarray(y_pred),
        }
    ).to_csv(path, index=False)


def save_feature_importance(path: Path, feature_cols: list[str], importances) -> None:
    values = np.asarray(importances)
    if values.ndim == 1:
        frame = pd.DataFrame({"feature": feature_cols, "importance": values})
        sort_column = "importance"
    else:
        if values.shape[0] != len(feature_cols) and values.shape[-1] == len(feature_cols):
            values = values.reshape(-1, len(feature_cols)).T
        elif values.shape[0] == len(feature_cols):
            values = values.reshape(len(feature_cols), -1)
        else:
            return

        columns = [f"importance_{index}" for index in range(values.shape[1])]
        frame = pd.DataFrame(values, columns=columns)
        frame.insert(0, "feature", feature_cols)
        frame["importance_mean"] = frame[columns].mean(axis=1)
        sort_column = "importance_mean"

    frame.sort_values(sort_column, ascending=False).to_csv(path, index=False)


def importance_column(frame: pd.DataFrame) -> str | None:
    for column in ("importance_mean", "importance"):
        if column in frame.columns:
            return column
    numeric_columns = [
        column for column in frame.columns
        if column != "feature" and pd.api.types.is_numeric_dtype(frame[column])
    ]
    if numeric_columns:
        frame["importance_mean"] = frame[numeric_columns].mean(axis=1)
        return "importance_mean"
    return None


def pearson_r(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def draw_prediction_correlation_axes(
    axes,
    results: list[dict],
    comparison_dir: Path,
    *,
    title_fontsize: int = 12,
    label_fontsize: int = 11,
    tick_fontsize: int = 10,
) -> None:
    for ax, result in zip(axes, results):
        model_name = result["model"]
        prediction_path = comparison_dir / model_name / "test_predictions.csv"
        if not prediction_path.exists():
            ax.text(0.5, 0.5, "missing predictions", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue

        predictions = pd.read_csv(prediction_path)
        ax.scatter(
            predictions["y_true"],
            predictions["y_pred"],
            s=40,
            alpha=0.72,
            color="#2563eb",
            edgecolors="#1f2937",
            linewidths=0.35,
        )
        min_value = min(predictions["y_true"].min(), predictions["y_pred"].min())
        max_value = max(predictions["y_true"].max(), predictions["y_pred"].max())
        ax.plot([min_value, max_value], [min_value, max_value], color="#dc2626", linestyle="--", linewidth=1.4)
        ax.set_title(
            MODEL_LABELS.get(model_name, model_name),
            fontsize=title_fontsize,
            weight="bold",
            pad=8,
        )
        ax.set_xlabel("True log10(conductivity)", fontsize=label_fontsize)
        ax.set_ylabel("Predicted log10(conductivity)", fontsize=label_fontsize)
        ax.tick_params(axis="both", labelsize=tick_fontsize)
        ax.grid(alpha=0.25)

    for ax in axes[len(results):]:
        ax.set_axis_off()


def plot_prediction_correlation(results: list[dict], comparison_dir: Path) -> None:
    if not results:
        return

    fig_dir = comparison_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(results), figsize=(5.0 * len(results), 4.5), squeeze=False)

    draw_prediction_correlation_axes(axes[0], results, comparison_dir)
    fig.suptitle("Test Set Predicted vs True Values", fontsize=16, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(fig_dir / "predicted_vs_true_all_models.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def lightgbm_top10_feature_share(comparison_dir: Path) -> pd.DataFrame | None:
    importance_path = comparison_dir / "lightgbm" / "feature_importance.csv"
    if not importance_path.exists():
        print("LightGBM feature importance is missing; skipping top10 feature plot.")
        return None

    frame = pd.read_csv(importance_path)
    column = importance_column(frame)
    if column is None:
        print("No numeric LightGBM feature importance column found; skipping top10 feature plot.")
        return None

    frame = frame.sort_values(column, ascending=False).copy()
    total_importance = frame[column].sum()
    if total_importance <= 0:
        print("LightGBM feature importance sums to zero; skipping top10 feature plot.")
        return None

    top10 = frame.head(10).copy()
    top10["importance_share"] = top10[column] / total_importance
    top10[["feature", column, "importance_share"]].to_csv(
        comparison_dir / "lightgbm_top10_feature_share.csv",
        index=False,
    )
    return top10.sort_values("importance_share", ascending=True)


def draw_lightgbm_top10_feature_axis(
    ax,
    plot_data: pd.DataFrame | None,
    *,
    title_fontsize: int = 17,
    label_fontsize: int = 13,
    tick_fontsize: int = 12,
    value_fontsize: int = 12,
    title_pad: int = 10,
) -> None:
    if plot_data is None or plot_data.empty:
        ax.text(0.5, 0.5, "missing LightGBM feature importance", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    values = plot_data["importance_share"] * 100
    labels = [display_feature_label(feature) for feature in plot_data["feature"]]
    positions = np.arange(len(plot_data))
    ax.barh(positions, values, color="#0f766e", alpha=0.92)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Share of total LightGBM importance (%)", fontsize=label_fontsize)
    ax.set_title("LightGBM Top 10 Feature Importance Share", fontsize=title_fontsize, weight="bold", pad=title_pad)
    ax.tick_params(axis="x", labelsize=tick_fontsize)
    ax.tick_params(axis="y", labelsize=tick_fontsize, pad=8)
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    ax.margins(y=0.05)
    ax.set_xlim(0, values.max() * 1.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for index, value in enumerate(values):
        ax.text(value, index, f" {value:.1f}%", va="center", fontsize=value_fontsize)


def plot_lightgbm_top10_feature_share(comparison_dir: Path) -> None:
    plot_data = lightgbm_top10_feature_share(comparison_dir)
    if plot_data is None:
        return

    fig_dir = comparison_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10.5, 7))
    draw_lightgbm_top10_feature_axis(ax, plot_data)
    fig.tight_layout()
    fig.savefig(fig_dir / "lightgbm_top10_feature_share.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def metric_table_frame(comparison: list[dict]) -> pd.DataFrame:
    columns = [
        "model",
        "train_mae",
        "test_mae",
        "train_rmse",
        "test_rmse",
        "train_r2",
        "test_r2",
    ]
    frame = pd.DataFrame(comparison)[columns].copy()
    frame["model"] = frame["model"].map(lambda value: MODEL_LABELS.get(value, value))
    for column in columns[1:]:
        frame[column] = frame[column].map(lambda value: f"{value:.3f}")
    return frame


def draw_metric_table_axis(
    ax,
    frame: pd.DataFrame,
    *,
    title_fontsize: int = 14,
    table_fontsize: int = 9,
    y_scale: float = 1.35,
    title_pad: int = 6,
    table_bbox: list[float] | None = None,
) -> None:
    if frame.empty:
        ax.text(0.5, 0.5, "missing metrics", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    ax.axis("off")
    if table_bbox is None:
        table_bbox = [0.0, 0.05, 1.0, 0.82]

    table = ax.table(
        cellText=frame.values,
        colLabels=["Model", "Train MAE", "Test MAE", "Train RMSE", "Test RMSE", "Train R²", "Test R²"],
        cellLoc="center",
        colLoc="center",
        loc="upper center",
        bbox=table_bbox,
        colWidths=[0.18, 0.136, 0.136, 0.136, 0.136, 0.136, 0.136],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(table_fontsize)
    table.scale(1, y_scale)
    highlight_rows = set(frame.index[frame["model"] == "LightGBM"] + 1)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#d1d5db")
        cell.set_linewidth(0.7)
        if row == 0:
            cell.set_facecolor("#1f2937")
            cell.set_text_props(color="white", weight="bold", fontsize=table_fontsize)
            cell.set_linewidth(0.9)
        elif row in highlight_rows:
            cell.set_facecolor("#f3f4f6" if row % 2 == 0 else "#ffffff")
            cell.set_text_props(color="black", weight="bold", fontsize=table_fontsize)
        elif row % 2 == 0:
            cell.set_facecolor("#f3f4f6")
        else:
            cell.set_facecolor("#ffffff")

    ax.set_title("Train/Test Metrics Across Models", fontsize=title_fontsize, weight="bold", pad=title_pad)


def plot_metric_table(comparison: list[dict], comparison_dir: Path) -> None:
    if not comparison:
        return

    fig_dir = comparison_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    frame = metric_table_frame(comparison)
    fig, ax = plt.subplots(figsize=(11.5, 3.8 + 0.35 * len(frame)))
    draw_metric_table_axis(ax, frame, table_bbox=[0.0, 0.05, 1.0, 0.84])
    fig.tight_layout()
    fig.savefig(fig_dir / "train_test_metrics_table.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_combined_training_summary(results: list[dict], comparison: list[dict], comparison_dir: Path) -> None:
    if not results and not comparison:
        return

    fig_dir = comparison_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    model_count = max(1, len(results))
    fig_width = max(18.0, 4.8 * model_count)
    fig = plt.figure(figsize=(fig_width, 11.4), constrained_layout=False)
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.0, 1.12],
        width_ratios=[2.0, 1.0],
        hspace=0.31,
        wspace=0.34,
    )

    top_grid = grid[0, :].subgridspec(1, model_count, wspace=0.28)
    prediction_axes = [fig.add_subplot(top_grid[0, index]) for index in range(model_count)]
    draw_prediction_correlation_axes(
        prediction_axes,
        results,
        comparison_dir,
        title_fontsize=14,
        label_fontsize=12,
        tick_fontsize=11,
    )

    table_ax = fig.add_subplot(grid[1, 0])
    feature_grid = grid[1, 1].subgridspec(2, 1, height_ratios=[0.86, 0.14], hspace=0.0)
    feature_ax = fig.add_subplot(feature_grid[0, 0])
    draw_metric_table_axis(
        table_ax,
        metric_table_frame(comparison) if comparison else pd.DataFrame(),
        title_fontsize=16,
        table_fontsize=11,
        y_scale=1.55,
        title_pad=4,
        table_bbox=[0.0, 0.12, 1.0, 0.76],
    )
    draw_lightgbm_top10_feature_axis(
        feature_ax,
        lightgbm_top10_feature_share(comparison_dir),
        title_fontsize=13,
        label_fontsize=11,
        tick_fontsize=9,
        value_fontsize=10,
        title_pad=6,
    )

    fig.suptitle("Model Performance Summary", fontsize=20, weight="bold", y=0.98)
    fig.savefig(fig_dir / "model_performance_summary_combined.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_training_summary(results: list[dict], comparison: list[dict], comparison_dir: Path) -> None:
    plot_prediction_correlation(results, comparison_dir)
    plot_lightgbm_top10_feature_share(comparison_dir)
    plot_metric_table(comparison, comparison_dir)
    plot_combined_training_summary(results, comparison, comparison_dir)
    print(f"Summary figures saved to {comparison_dir / 'figures'}")


def cv_evaluate_model(model_factory, X, y, w, configs, *, use_weight=True, scale=False):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    best = None
    results = []

    for index, config in enumerate(configs, start=1):
        fold_maes = []
        for train_idx, valid_idx in kf.split(X):
            X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
            y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
            w_train = w.iloc[train_idx] if use_weight else None

            if scale:
                scaler = StandardScaler()
                X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
                X_valid = pd.DataFrame(scaler.transform(X_valid), columns=X.columns)

            model = model_factory(config)
            if use_weight:
                try:
                    model.fit(X_train, y_train, sample_weight=w_train)
                except TypeError:
                    model.fit(X_train, y_train)
            else:
                model.fit(X_train, y_train)

            pred = model.predict(X_valid)
            fold_maes.append(mean_absolute_error(y_valid, pred))

        result = {
            "params": config,
            "mae_mean": float(np.mean(fold_maes)),
            "mae_std": float(np.std(fold_maes)),
        }
        results.append(result)
        marker = ""
        if best is None or result["mae_mean"] < best["mae_mean"]:
            best = result
            marker = " *** BEST ***"
        print(f"  [{index:2d}] CV MAE={result['mae_mean']:.4f}±{result['mae_std']:.4f}{marker}")

    return best, results


def fit_final(model, X_train, y_train, w_train, X_test, y_test, *, use_weight=True, scale=False):
    scaler = None
    X_train_fit = X_train
    X_test_fit = X_test
    if scale:
        scaler = StandardScaler()
        X_train_fit = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
        X_test_fit = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

    if use_weight:
        try:
            model.fit(X_train_fit, y_train, sample_weight=w_train)
        except TypeError:
            model.fit(X_train_fit, y_train)
    else:
        model.fit(X_train_fit, y_train)

    train_pred = model.predict(X_train_fit)
    test_pred = model.predict(X_test_fit)
    return model, scaler, train_pred, test_pred


def train_lightgbm(data: dict, args: argparse.Namespace) -> dict:
    configs = [
        {"n_estimators": 1000, "learning_rate": 0.03, "num_leaves": 15, "min_child_samples": 10, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0, "reg_alpha": 0.0, "max_depth": -1},
        {"n_estimators": 1000, "learning_rate": 0.05, "num_leaves": 15, "min_child_samples": 15, "subsample": 0.8, "colsample_bytree": 0.6, "reg_lambda": 5.0, "reg_alpha": 1.0, "max_depth": 7},
        {"n_estimators": 1000, "learning_rate": 0.05, "num_leaves": 23, "min_child_samples": 10, "subsample": 0.7, "colsample_bytree": 0.6, "reg_lambda": 10.0, "reg_alpha": 2.0, "max_depth": 5},
        {"n_estimators": 1000, "learning_rate": 0.08, "num_leaves": 15, "min_child_samples": 20, "subsample": 0.8, "colsample_bytree": 0.5, "reg_lambda": 10.0, "reg_alpha": 2.0, "max_depth": 5},
        {"n_estimators": 500, "learning_rate": 0.1, "num_leaves": 7, "min_child_samples": 30, "subsample": 0.8, "colsample_bytree": 0.7, "reg_lambda": 10.0, "reg_alpha": 5.0, "max_depth": 3},
    ]

    def factory(config):
        return lgb.LGBMRegressor(
            objective="regression",
            verbosity=-1,
            n_jobs=-1,
            random_state=42,
            subsample_freq=1,
            **config,
        )

    return train_from_configs("lightgbm", factory, configs, data, args, use_weight=True)


def train_random_forest(data: dict, args: argparse.Namespace) -> dict:
    configs = [
        {"n_estimators": 500, "max_depth": None, "min_samples_split": 5, "min_samples_leaf": 2, "max_features": "sqrt"},
        {"n_estimators": 500, "max_depth": 15, "min_samples_split": 5, "min_samples_leaf": 2, "max_features": 0.7},
        {"n_estimators": 500, "max_depth": 10, "min_samples_split": 10, "min_samples_leaf": 5, "max_features": 0.6},
        {"n_estimators": 500, "max_depth": 20, "min_samples_split": 5, "min_samples_leaf": 3, "max_features": 0.8},
        {"n_estimators": 1000, "max_depth": 15, "min_samples_split": 5, "min_samples_leaf": 2, "max_features": "sqrt"},
    ]
    return train_from_configs(
        "random_forest",
        lambda config: RandomForestRegressor(random_state=42, n_jobs=-1, **config),
        configs,
        data,
        args,
        use_weight=True,
    )


def train_decision_tree(data: dict, args: argparse.Namespace) -> dict:
    configs = [
        {"max_depth": 5, "min_samples_split": 10, "min_samples_leaf": 5},
        {"max_depth": 7, "min_samples_split": 10, "min_samples_leaf": 5},
        {"max_depth": 10, "min_samples_split": 10, "min_samples_leaf": 5},
        {"max_depth": None, "min_samples_split": 10, "min_samples_leaf": 5},
    ]
    return train_from_configs(
        "decision_tree",
        lambda config: DecisionTreeRegressor(random_state=42, **config),
        configs,
        data,
        args,
        use_weight=True,
    )


def train_mlp(data: dict, args: argparse.Namespace) -> dict:
    configs = [
        {"hidden_layer_sizes": (64, 32), "alpha": 0.01, "learning_rate_init": 0.001, "max_iter": 1000},
        {"hidden_layer_sizes": (128, 64), "alpha": 0.01, "learning_rate_init": 0.001, "max_iter": 1000},
        {"hidden_layer_sizes": (64, 32), "alpha": 0.1, "learning_rate_init": 0.001, "max_iter": 1000},
        {"hidden_layer_sizes": (64,), "alpha": 0.01, "learning_rate_init": 0.001, "max_iter": 1000},
    ]
    return train_from_configs(
        "mlp",
        lambda config: MLPRegressor(random_state=42, early_stopping=True, validation_fraction=0.1, **config),
        configs,
        data,
        args,
        use_weight=False,
        scale=True,
    )


def train_ngboost(data: dict, args: argparse.Namespace) -> dict | None:
    try:
        from ngboost import NGBRegressor
        from ngboost.distns import Normal
    except ImportError:
        print("NGBoost is not installed; skipping.")
        return None

    configs = [
        {"n_estimators": 500, "learning_rate": 0.01, "minibatch_frac": 0.8},
        {"n_estimators": 500, "learning_rate": 0.05, "minibatch_frac": 0.8},
        {"n_estimators": 300, "learning_rate": 0.05, "minibatch_frac": 1.0},
        {"n_estimators": 500, "learning_rate": 0.03, "minibatch_frac": 0.9},
    ]
    return train_from_configs(
        "ngboost",
        lambda config: NGBRegressor(Dist=Normal, random_state=42, verbose=False, **config),
        configs,
        data,
        args,
        use_weight=False,
    )


def train_from_configs(model_name, factory, configs, data, args, *, use_weight=True, scale=False) -> dict:
    print("\n" + "=" * 60)
    print(model_name.upper())
    print("=" * 60)
    start = time.time()
    best, cv_results = cv_evaluate_model(
        factory,
        data["X_train"],
        data["y_train"],
        data["w_train"],
        configs,
        use_weight=use_weight,
        scale=scale,
    )
    model, scaler, train_pred, test_pred = fit_final(
        factory(best["params"]),
        data["X_train"],
        data["y_train"],
        data["w_train"],
        data["X_test"],
        data["y_test"],
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

    result = {
        "model": model_name,
        "feature_set": feature_set_name(args.train),
        "train_path": str(args.train),
        "test_path": str(args.test),
        "n_features": len(data["feature_cols"]),
        "best_params": best["params"],
        "cv_best_mae": best["mae_mean"],
        "cv_best_mae_std": best["mae_std"],
        "cv_results": cv_results,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "elapsed_seconds": elapsed,
        "upper_bound_replacement": args.upper_bound_replacement,
        "upper_bound_weight": args.upper_bound_weight,
    }
    save_json(model_dir / "final_results.json", result)

    print(f"Best params: {best['params']}")
    print(f"Train: MAE={train_metrics['mae']:.4f}, RMSE={train_metrics['rmse']:.4f}, R2={train_metrics['r2']:.4f}")
    print(f"Test:  MAE={test_metrics['mae']:.4f}, RMSE={test_metrics['rmse']:.4f}, R2={test_metrics['r2']:.4f}")
    print(f"Outputs: {model_dir}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train models on a selected feature split.")
    parser.add_argument("--model", choices=["lightgbm", "rf", "dt", "mlp", "ngboost", "all"], default="all")
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--no-interactions", action="store_true")
    parser.add_argument("--remove-upper-bound", action="store_true")
    parser.add_argument("--upper-bound-threshold", type=float, default=1e-10)
    parser.add_argument("--upper-bound-replacement", type=float, default=1e-11)
    parser.add_argument("--upper-bound-weight", type=float, default=0.3)
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
