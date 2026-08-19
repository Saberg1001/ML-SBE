"""Train the V2 absolute-conductivity LightGBM model with F35 + two features."""

from __future__ import annotations

import json
import os
import sys

if __package__ is None:
    _PROJECT_ROOT = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

from pathlib import Path

import pandas as pd

from main.absolute.split import SplitConfig, split_feature_table
from main.absolute.train import TrainConfig, train_model
from main.features import FeatureConfig, make_feature_table
from main.paths import DATA_DIR, RUNS_DIR


INPUT_PATH = DATA_DIR / "absolute" / "data-absolute-v2-model-clean.csv"
RUN_NAME = "abs_v2_f37_native_family_lgbm_trials50_seed42"


def main() -> None:
    """Build F37 from the V2 clean table and train only native-family LightGBM."""

    source = pd.read_csv(INPUT_PATH)
    feature_config = FeatureConfig(
        min_conductivity=None,
        include_family=True,
        family_encoding="native",
        include_interactions=True,
        include_small_features=True,
        drop_redundant=True,
        output_path=None,
    )
    feature_result = make_feature_table(source, feature_config)
    split_result = split_feature_table(
        feature_result.table,
        SplitConfig(method="random", test_size=0.2, seed=42),
    )
    train_result = train_model(
        split_result.train,
        split_result.test,
        TrainConfig(
            model_name="lightgbm",
            n_trials=50,
            cv_splits=5,
            seed=42,
            optuna_seed=42,
            output_root=RUNS_DIR / "absolute",
            run_name=RUN_NAME,
            dataset_name="absolute_v2_clean_f37_native_family",
            categorical_features=["family"],
            verbose=True,
        ),
    )
    output_dir = train_result.output_dir
    feature_result.table.to_csv(output_dir / "data" / "all_features.csv", index=False)
    (output_dir / "feature_build_summary.json").write_text(
        json.dumps(feature_result.summary, ensure_ascii=False, indent=2, default=str)
        + "\n",
        encoding="utf-8",
    )
    print(f"Input rows   : {len(source)}")
    print(f"Feature rows : {len(feature_result.table)}")
    print(f"Features     : {len(feature_result.feature_columns)}")
    print(f"Output dir   : {output_dir.resolve()}")


if __name__ == "__main__":
    main()
