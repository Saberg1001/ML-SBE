from __future__ import annotations

import os
import sys

# Allow running this module directly with ``python main/absolute/pipeline.py``
# (e.g. the VS Code "Run" button): project root must be importable for ``main.*``.
if __package__ is None:
    _PROJECT_ROOT = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

from dataclasses import dataclass, field

from main.absolute.data import CleanDataConfig, CleanDataResult, clean_raw_data
from main.absolute.split import SplitConfig, SplitResult, split_feature_table
from main.absolute.train import TrainConfig, TrainResult, train_model
from main.features import FeatureConfig, FeatureResult, make_feature_table
from main.paths import RUNS_DIR


@dataclass
class PipelineConfig:
    """Top-level configuration for the full training workflow.

    clean:
        CleanDataConfig for raw data loading and row-level cleaning.
    features:
        FeatureConfig for conductivity filtering and descriptor construction.
    split:
        SplitConfig for train/test creation.
    train:
        TrainConfig for model selection, Optuna tuning, and output packaging.
    """

    clean: CleanDataConfig = field(default_factory=CleanDataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


@dataclass
class PipelineResult:
    clean: CleanDataResult
    features: FeatureResult
    split: SplitResult
    train: TrainResult


# Default training model used when one is not specified explicitly.
# Supported: "all" | "lightgbm" | "random_forest" | "decision_tree" | "mlp" | "ngboost"
DEFAULT_TRAIN_MODEL = "all"


def default_pipeline_config() -> PipelineConfig:
    """Single place holding all tunable absolute-training parameters.

    To tune the absolute-value regression run, edit values in this function
    instead of the individual sub-config classes; their own defaults are kept
    only as fallbacks for direct stage calls.
    """

    return PipelineConfig(
        clean=CleanDataConfig(
            remove_organic=True,
            remove_charge_abnormal=True,
            charge_residual_limit=6.0,
        ),
        features=FeatureConfig(
            min_conductivity=1e-6,
            include_family=True,
            include_interactions=True,
            include_small_features=True,
            drop_redundant=True,
            output_path=None,
        ),
        split=SplitConfig(
            method="random",
            test_size=0.2,
            seed=42,
        ),
        train=TrainConfig(
            model_name=DEFAULT_TRAIN_MODEL,
            n_trials=50,
            cv_splits=5,
            seed=42,
            optuna_seed=42,
            output_root=RUNS_DIR / "absolute",
            dataset_name="ionic_main_absolute",
            verbose=True,
        ),
    )


def run_training_pipeline(config: PipelineConfig | None = None) -> PipelineResult:
    """Run clean -> feature generation -> split -> train with one config object.

    When ``config`` is omitted, the default settings in
    :func:`default_pipeline_config` are used (see that function to tune runs).
    """

    config = config or default_pipeline_config()
    clean_result = clean_raw_data(config=config.clean)
    feature_result = make_feature_table(clean_result.cleaned, config.features)
    split_result = split_feature_table(feature_result.table, config.split)
    train_result = train_model(split_result.train, split_result.test, config.train)
    return PipelineResult(
        clean=clean_result,
        features=feature_result,
        split=split_result,
        train=train_result,
    )


def main() -> None:
    """Run the entire absolute-value training pipeline with defaults.

    Run directly via ``python main/absolute/pipeline.py`` (or the VS Code "Run" button).
    This is the single entry point that chains clean -> features -> split ->
    train using :func:`default_pipeline_config`.

    Note: the default configuration trains all models with 50 Optuna trials
    per model; this step can take a while.

    Intermediate outputs : data/modeling/absolute/generated_{clean,features,
                           train,test}.csv
    Final output         : runs/absolute/<run_id>/
                           (model_comparison.csv, per-model predictions,
                           figures/, config.json, summary.json, best_model.txt)
    """
    result = run_training_pipeline()
    print(f"Best model: {result.train.best_model}")
    print(f"Final output dir: {result.train.output_dir.resolve()}")


if __name__ == "__main__":
    main()
