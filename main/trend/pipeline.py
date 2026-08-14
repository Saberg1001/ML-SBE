"""Centralized configuration for the trend prediction workflow.

Mirrors ``main/absolute/pipeline.py::default_pipeline_config()``: all tunable
trend-prediction parameters live here in one place instead of inline argparse
defaults sprinkled across scripts.

The reusable entry point is :func:`main.trend.predict.predict_trend`, which
consumes :func:`default_trend_predict_config`.
"""

from __future__ import annotations

import os
import sys

# Allow running this module directly as a script (e.g. VS Code "Run" button):
# project root must be importable for ``main.*``.
if __package__ is None:
    _PROJECT_ROOT = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

from dataclasses import dataclass
from pathlib import Path

from main.paths import EXPERIMENTAL_ANNOTATIONS_DIR, EXPERIMENTAL_RAW_DIR, RUNS_DIR
from main.trend.features import TREND_ABSOLUTE_THRESHOLD_S_CM


@dataclass(frozen=True)
class TrendPredictConfig:
    """Options for the reusable trend-classification prediction run.

    All paths are fixed by ``default_trend_predict_config()``; edit there if
    you need to change inputs, model, or output location.
    """

    raw_csv: Path
    annotations: Path
    model: Path
    output: Path
    metrics: Path
    raw_scale: float = 1.0
    threshold_s_cm: float = TREND_ABSOLUTE_THRESHOLD_S_CM
    class_labels: tuple = ("decrease", "unchanged", "increase")


def default_trend_predict_config() -> TrendPredictConfig:
    """Single place holding the default trend-prediction parameters."""
    return TrendPredictConfig(
        raw_csv=EXPERIMENTAL_RAW_DIR / "experimental-data.csv",
        annotations=EXPERIMENTAL_ANNOTATIONS_DIR / "experimental-data-labeled.csv",
        model=RUNS_DIR
        / "trend"
        / "trend_cls_v3_f27_family_abs01_fixedval_optuna10_seed42"
        / "catboost"
        / "model.joblib",
        output=EXPERIMENTAL_ANNOTATIONS_DIR / "experimental-data-predict-trend.csv",
        metrics=EXPERIMENTAL_ANNOTATIONS_DIR / "experimental-data-predict-trend-metrics.json",
        raw_scale=1.0,
        threshold_s_cm=TREND_ABSOLUTE_THRESHOLD_S_CM,
        class_labels=("decrease", "unchanged", "increase"),
    )
