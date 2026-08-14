"""Absolute-prediction pipeline (clean -> features -> split -> train -> predict).

This subpackage groups the absolute-conductivity regression workflow. It shares
the feature engine (composition_features, normalize_family, ...) and path
constants with the rest of the project via the top-level modules:

- main.features : shared composition-feature engine + absolute feature table
- main.paths    : shared filesystem layout

Each staged module exposes a no-argument ``main()`` runnable from VS Code
("Run" button) that reads its fixed default input and writes to its default
output under data/modeling/absolute/ (or runs/absolute/ for training).
"""

from .data import CleanDataConfig, CleanDataResult, clean_raw_data, load_raw_data
from .pipeline import PipelineConfig, PipelineResult, run_training_pipeline
from .predict import PredictConfig, PredictionResult, label_families_by_blocks, predict_formulas
from .split import SplitConfig, SplitResult, split_feature_table
from .train import TrainConfig, TrainResult, train_model

__all__ = [
    "CleanDataConfig",
    "CleanDataResult",
    "PipelineConfig",
    "PipelineResult",
    "PredictConfig",
    "PredictionResult",
    "SplitConfig",
    "SplitResult",
    "TrainConfig",
    "TrainResult",
    "clean_raw_data",
    "load_raw_data",
    "label_families_by_blocks",
    "predict_formulas",
    "run_training_pipeline",
    "split_feature_table",
    "train_model",
]