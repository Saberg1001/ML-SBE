"""Screen ALL absolute-conductivity models on the V2 F37 dataset.

Same 826-row v2 clean table, same F37 descriptors, same random split
(seed=42, test_size=0.2) and same tuning budget (50 Optuna trials, 5-fold CV)
as example one. To make the comparison fair across models that cannot consume a
native pandas categorical, the family label is ordinal-encoded for ALL five
models (this mirrors the v1 all-models screening). The deployed model in
example one additionally uses LightGBM's native categorical family encoding,
which is a LightGBM-only refinement applied after this screening.
"""

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
RUN_NAME = "abs_v2_f37_ordinal_family_all_models_trials50_seed42"
# Reuse the descriptor cache from the lgbm-only run so feature values (hence the
# split) are byte-identical.
CACHE_PATH = (
    RUNS_DIR
    / "absolute"
    / "abs_v2_f37_native_family_lgbm_trials50_seed42"
    / "data"
    / "all_features.csv"
)


def main() -> None:
    source = pd.read_csv(INPUT_PATH)
    feature_config = FeatureConfig(
        min_conductivity=None,
        include_family=True,
        family_encoding="ordinal",
        include_interactions=True,
        include_small_features=True,
        drop_redundant=True,
        output_path=None,
        descriptor_cache_path=CACHE_PATH,
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
            model_name="all",
            n_trials=50,
            cv_splits=5,
            seed=42,
            optuna_seed=42,
            output_root=RUNS_DIR / "absolute",
            run_name=RUN_NAME,
            dataset_name="absolute_v2_clean_f37_ordinal_family",
            categorical_features=None,
            n_jobs=1,
            verbose=True,
        ),
    )
    output_dir = train_result.output_dir
    print(f"Input rows   : {len(source)}")
    print(f"Feature rows : {len(feature_result.table)}")
    print(f"Features     : {len(feature_result.feature_columns)}")
    print(f"Output dir   : {output_dir.resolve()}")


if __name__ == "__main__":
    main()
