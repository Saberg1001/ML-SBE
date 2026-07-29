from __future__ import annotations

from dataclasses import dataclass, field

from .data import CleanDataConfig, CleanDataResult, clean_raw_data
from .features import FeatureConfig, FeatureResult, make_feature_table
from .split import SplitConfig, SplitResult, split_feature_table
from .train import TrainConfig, TrainResult, train_model


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


def run_training_pipeline(config: PipelineConfig | None = None) -> PipelineResult:
    """Run clean -> feature generation -> split -> train with one config object."""

    config = config or PipelineConfig()
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
