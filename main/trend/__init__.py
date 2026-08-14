"""Trend-classification pipeline (pair -> feature -> split -> train -> predict).

Reusable library/entry points for the conductivity *trend* (increase / unchanged /
decrease) modeling. Shares the composition feature engine with the absolute
regression pipeline via the top-level :mod:`main.features`.
"""

from __future__ import annotations

from .features import (
    B_BASELINE_FEATURES,
    MODEL_FEATURE_COLUMNS,
    SIGNED_DELTA_FEATURES,
    TREND_ABSOLUTE_THRESHOLD_S_CM,
    build_pair_feature_table,
    build_simple_pair_feature_table,
    build_prediction_features,
    classify_trend_delta,
    compute_formula_descriptor,
    feature_schema,
)
from .pipeline import TrendPredictConfig, default_trend_predict_config
from .predict import predict_trend
from .split import (
    DEFAULT_TRAIN,
    DEFAULT_VALIDATION,
    SPLIT_VERSION,
    split_feature_file,
)

__all__ = [
    "B_BASELINE_FEATURES",
    "MODEL_FEATURE_COLUMNS",
    "SIGNED_DELTA_FEATURES",
    "TREND_ABSOLUTE_THRESHOLD_S_CM",
    "TrendPredictConfig",
    "DEFAULT_TRAIN",
    "DEFAULT_VALIDATION",
    "SPLIT_VERSION",
    "build_pair_feature_table",
    "build_prediction_features",
    "build_simple_pair_feature_table",
    "classify_trend_delta",
    "compute_formula_descriptor",
    "default_trend_predict_config",
    "feature_schema",
    "predict_trend",
    "split_feature_file",
]
