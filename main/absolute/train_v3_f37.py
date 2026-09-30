"""Train the V3 absolute-conductivity models (F37) with predictive uncertainty.

Data
----
data/absolute/data-absolute-v3-model-clean.csv (main/absolute/clean_v3.py):
v2 + 21 curated halides + 236 sulfides, cleaned with the v2 policy plus the
room-temperature window on sulfide rows. Same F37 descriptors, same random
split (seed=42, test_size=0.2) and tuning budget (50 Optuna trials, 5-fold CV)
as the v2 runs so v2 -> v3 differences come from the data alone.

Uncertainty
-----------
Every model writes ``y_sigma`` (log10 space) next to ``y_pred`` and reports
coverage / z-score calibration in final_results.json and model_comparison.csv:
  * NGBoost: native Normal ``scale`` from ``pred_dist`` (learned per-row).
  * LightGBM / RF / DT / MLP: LightGBM has no variance output of its own, so a
    bootstrap ensemble of ``--n-bootstrap`` refits with the tuned params gives
    the epistemic spread, and the out-of-fold residual std of the same params
    is added as an aleatoric floor. The ensemble is saved in model.joblib so
    predict.py reports sigma for these models too.

Usage
-----
    python main/absolute/train_v3_f37.py                    # all five models
    python main/absolute/train_v3_f37.py --model lightgbm   # native-family LightGBM only
    python main/absolute/train_v3_f37.py --n-bootstrap 0    # skip bootstrap sigma
"""

from __future__ import annotations

import argparse
import json
import os
import sys

if __package__ is None:
    _PROJECT_ROOT = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

import pandas as pd

from main.absolute.split import SplitConfig, split_feature_table
from main.absolute.train import MODEL_NAMES, TrainConfig, train_model
from main.features import FeatureConfig, make_feature_table
from main.paths import DATA_DIR, RUNS_DIR


INPUT_PATH = DATA_DIR / "absolute" / "data-absolute-v3-model-clean.csv"
# Descriptor cache shared by every v3 run so feature values (hence the split)
# are byte-identical across model selections. Seeded from the v2 cache on the
# first run (the ~800 v2 formulas need no recomputation), then written back.
CACHE_PATH = RUNS_DIR / "absolute" / "abs_v3_f37_feature_cache" / "all_features.csv"
V2_CACHE_PATH = (
    RUNS_DIR / "absolute" / "abs_v2_f37_native_family_lgbm_trials50_seed42" / "data" / "all_features.csv"
)


def _run_name(model: str, family_encoding: str, n_trials: int, seed: int) -> str:
    model_tag = "all_models" if model == "all" else model
    return f"abs_v3_f37_{family_encoding}_family_{model_tag}_trials{n_trials}_seed{seed}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="all", choices=["all", *MODEL_NAMES])
    parser.add_argument(
        "--family-encoding",
        default=None,
        choices=["native", "ordinal"],
        help="Default: native for a lightgbm-only run (as the deployed v2 lgbm), ordinal otherwise.",
    )
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--n-bootstrap", type=int, default=30, help="Bootstrap replicas for sigma on point models; 0 disables.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    family_encoding = args.family_encoding or ("native" if args.model == "lightgbm" else "ordinal")
    run_name = args.run_name or _run_name(args.model, family_encoding, args.n_trials, args.seed)

    source = pd.read_csv(INPUT_PATH)
    cache_path = CACHE_PATH if CACHE_PATH.exists() else (V2_CACHE_PATH if V2_CACHE_PATH.exists() else None)
    feature_config = FeatureConfig(
        min_conductivity=None,
        include_family=True,
        family_encoding=family_encoding,
        include_interactions=True,
        include_small_features=True,
        drop_redundant=True,
        output_path=None,
        descriptor_cache_path=cache_path,
    )
    feature_result = make_feature_table(source, feature_config)
    if not CACHE_PATH.exists():
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        feature_result.table.to_csv(CACHE_PATH, index=False)
    split_result = split_feature_table(
        feature_result.table,
        SplitConfig(method="random", test_size=0.2, seed=args.seed),
    )
    train_result = train_model(
        split_result.train,
        split_result.test,
        TrainConfig(
            model_name=args.model,
            n_trials=args.n_trials,
            cv_splits=5,
            seed=args.seed,
            optuna_seed=args.seed,
            output_root=RUNS_DIR / "absolute",
            run_name=run_name,
            dataset_name=f"absolute_v3_clean_f37_{family_encoding}_family",
            categorical_features=["family"] if family_encoding == "native" else None,
            n_jobs=args.n_jobs,
            n_bootstrap=args.n_bootstrap,
            bootstrap_seed=args.seed,
            verbose=True,
        ),
    )
    output_dir = train_result.output_dir
    feature_result.table.to_csv(output_dir / "data" / "all_features.csv", index=False)
    (output_dir / "feature_build_summary.json").write_text(
        json.dumps(feature_result.summary, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(f"Input rows   : {len(source)}")
    print(f"Feature rows : {len(feature_result.table)}")
    print(f"Features     : {len(feature_result.feature_columns)}")
    print(f"Best model   : {train_result.best_model}")
    print(f"Output dir   : {output_dir.resolve()}")
    print(train_result.comparison.to_string(index=False))


if __name__ == "__main__":
    main()
