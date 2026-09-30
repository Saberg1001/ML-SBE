"""Augment the V2 F37 absolute NGBoost model with the lab's ball-milled halides.

Goal
----
Improve prediction accuracy for THIS lab's ball-milled halides -- the region
where the deployed model extrapolated and mispredicted the Ca composition
(hal_023 = Li2.6In0.4Zr0.5Ca0.1Cl6, now measured at 0.59 mS/cm).

Design decisions (documented for auditability)
----------------------------------------------
- Data added: 13 clean-formula, ball-milled halides measured in-house
  (data/experimental/lab-halide-ballmill.csv). The 10 solid-state "+5wt%ZrCl4"
  rows are excluded: the additive confounds the synthesis-method offset and
  cannot be featurized cleanly.
- New feature: synth_method_code (literature/unknown=0, ball_mill=1), an ordinal
  passthrough so NGBoost can separate the ball-milled-halide offset from
  chemistry. At inference set method=ball_mill to get the offset-corrected mu.
- Evaluation: leave-one-out (LOO) over the 13 lab halides, three ways, to
  separate the value of *adding data* from the value of the *method feature*:
    B0  deployed lit-only model (existing run)  -> predict all 13, no retrain
    B1  LOO, lit + 12 lab halides, F37          -> value of adding data alone
    B2  LOO, lit + 12 lab halides, F38          -> value of data + method feature
  LOO reuses the deployed NGBoost hyperparameters (no per-fold Optuna).
- Deployment: a final NGBoost fit on ALL 826 lit + 13 lab halides with F38,
  reusing the deployed hyperparameters, plus a no-regression check on the
  standard literature test split (seed 42, 20%).
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

import joblib
import numpy as np
import pandas as pd

from main.absolute.split import SplitConfig, split_feature_table
from main.features import FeatureConfig, make_feature_table
from main.paths import DATA_DIR, RUNS_DIR

TARGET = "log10_conductivity"
LIT_INPUT = DATA_DIR / "absolute" / "data-absolute-v2-model-clean.csv"
LAB_INPUT = DATA_DIR / "experimental" / "lab-halide-ballmill.csv"
CACHE_PATH = (
    RUNS_DIR / "absolute"
    / "abs_v2_f37_native_family_lgbm_trials50_seed42" / "data" / "all_features.csv"
)
DEPLOYED_RUN = RUNS_DIR / "absolute" / "abs_v2_f37_ordinal_family_all_models_trials50_seed42"
OUT_RUN = RUNS_DIR / "absolute" / "abs_v2_f38_lab_halide_ballmill_ngboost"

# Deployed NGBoost hyperparameters (runs/.../ngboost/final_results.json).
NGB_PARAMS = {"n_estimators": 1200, "learning_rate": 0.03246185745266487, "minibatch_frac": 0.5193625999805914}
SYNTH_COL = "synth_method_code"
Z95 = 1.959963984540054


def _ngb_factory(params):
    from ngboost import NGBRegressor
    from ngboost.distns import Normal

    return NGBRegressor(Dist=Normal, random_state=42, verbose=False, **params)


def _medians(frame: pd.DataFrame, feature_columns: list[str]) -> pd.Series:
    matrix = (
        frame.reindex(columns=feature_columns)
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )
    return matrix.median().fillna(0.0)


def _matrix(frame: pd.DataFrame, feature_columns: list[str], medians: pd.Series) -> pd.DataFrame:
    matrix = (
        frame.reindex(columns=feature_columns)
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(medians)
    )
    return matrix


def _fit_predict(train: pd.DataFrame, held: pd.DataFrame, feature_columns: list[str]):
    medians = _medians(train, feature_columns)
    X_train = _matrix(train, feature_columns, medians).to_numpy()
    y_train = pd.to_numeric(train[TARGET], errors="coerce").to_numpy()
    X_held = _matrix(held, feature_columns, medians).to_numpy()
    model = _ngb_factory(NGB_PARAMS)
    model.fit(X_train, y_train)
    mu = model.predict(X_held)
    sigma = np.asarray(model.pred_dist(X_held).params["scale"], dtype=float)
    return mu, sigma


def _loo(lit: pd.DataFrame, lab: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    rows = []
    lab_ids = lab["ID"].tolist()
    for index, held_id in enumerate(lab_ids, start=1):
        held = lab[lab["ID"] == held_id]
        train = pd.concat([lit, lab[lab["ID"] != held_id]], ignore_index=True)
        mu, sigma = _fit_predict(train, held, feature_columns)
        rows.append(
            {
                "ID": held_id,
                "composition": held["True Composition"].iloc[0],
                "true": float(held[TARGET].iloc[0]),
                "mu": float(mu[0]),
                "sigma": float(sigma[0]),
            }
        )
        print(f"  LOO fold {index}/{len(lab_ids)} ({held_id}) done", flush=True)
    return pd.DataFrame(rows)


def _calibration(frame: pd.DataFrame) -> dict:
    residual = frame["true"] - frame["mu"]
    abs_err = residual.abs()
    inside = (abs_err <= Z95 * frame["sigma"]).mean()
    z = residual / frame["sigma"].replace(0, np.nan)
    return {
        "mae": float(abs_err.mean()),
        "rmse": float(np.sqrt((residual**2).mean())),
        "coverage95": float(inside),
        "z_std": float(z.std(ddof=0)),
        "z_mean": float(z.mean()),
    }


def main() -> None:
    # --- Featurize literature + lab together so family encoding is shared. ---
    lit_source = pd.read_csv(LIT_INPUT)
    lit_source["synth_method"] = "unknown"
    lab_source = pd.read_csv(LAB_INPUT)
    combined = pd.concat([lit_source, lab_source], ignore_index=True)

    config = FeatureConfig(
        min_conductivity=None,
        include_family=True,
        family_encoding="ordinal",
        include_interactions=True,
        include_small_features=True,
        drop_redundant=True,
        output_path=None,
        descriptor_cache_path=CACHE_PATH,
    )
    result = make_feature_table(combined, config)
    featured = result.table.reset_index(drop=True)
    f37 = list(result.feature_columns)

    # Ordinal synthesis-method feature (ball_mill=1, everything else=0).
    method_source = combined.set_index("ID")["synth_method"].fillna("unknown")
    method_code = featured["ID"].map(method_source).map(
        lambda value: 1 if str(value).strip() in ("ball_mill", "球磨") else 0
    )
    featured[SYNTH_COL] = pd.to_numeric(method_code, errors="coerce").fillna(0).astype(float)
    f38 = [*f37, SYNTH_COL]

    is_lab = featured["ID"].astype(str).str.startswith("hal_")
    lit_featured = featured[~is_lab].reset_index(drop=True)
    lab_featured = featured[is_lab].reset_index(drop=True)
    print(f"Literature rows: {len(lit_featured)} | lab halides: {len(lab_featured)} | F37={len(f37)} F38={len(f38)}")

    # --- B0: deployed lit-only model predicts the 13 lab halides (no retrain). ---
    from main.absolute.predict import PredictConfig, predict_formulas

    b0 = predict_formulas(
        LAB_INPUT, DEPLOYED_RUN,
        PredictConfig(model_name="ngboost", output_dir=OUT_RUN / "b0_deployed_on_lab"),
    ).predictions
    b0 = b0.rename(columns={"pred_log10_conductivity": "mu", "pred_log10_sigma": "sigma"})
    b0["true"] = np.log10(pd.to_numeric(lab_source.set_index("ID").loc[b0["ID"], "Ionic conductivity (S cm-1)"].to_numpy()))
    b0_frame = b0[["ID", "mu", "sigma", "true"]].copy()

    # --- B1 / B2: leave-one-out over the lab halides. ---
    print("Running B1 (F37, add data only)...", flush=True)
    b1 = _loo(lit_featured, lab_featured, f37)
    print("Running B2 (F38, data + method feature)...", flush=True)
    b2 = _loo(lit_featured, lab_featured, f38)

    metrics = {
        "B0_deployed_lit_only": _calibration(b0_frame),
        "B1_loo_add_data_f37": _calibration(b1),
        "B2_loo_data_plus_method_f38": _calibration(b2),
    }

    # --- Final deployable F38 model on all lit + all lab halides. ---
    lit_split = split_feature_table(lit_featured, SplitConfig(method="random", test_size=0.2, seed=42))
    aug_train = pd.concat([lit_split.train, lab_featured], ignore_index=True)
    medians = _medians(aug_train, f38)
    X_train = _matrix(aug_train, f38, medians).to_numpy()
    y_train = pd.to_numeric(aug_train[TARGET], errors="coerce").to_numpy()
    final_model = _ngb_factory(NGB_PARAMS)
    final_model.fit(X_train, y_train)

    lit_test = lit_split.test
    X_test = _matrix(lit_test, f38, medians).to_numpy()
    y_test = pd.to_numeric(lit_test[TARGET], errors="coerce").to_numpy()
    lit_test_mae = float(np.mean(np.abs(final_model.predict(X_test) - y_test)))
    metrics["final_literature_test_mae"] = lit_test_mae
    metrics["final_literature_test_rows"] = int(len(lit_test))

    model_dir = OUT_RUN / "ngboost"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": final_model,
            "scaler": None,
            "feature_cols": f38,
            "feature_medians": medians,
            "family_mapping": result.family_mapping,
            "categorical_features": [],
            "category_levels": {},
            "family_onehot_categories": [],
        },
        model_dir / "model.joblib",
    )
    (OUT_RUN / "best_model.txt").write_text("ngboost\n", encoding="utf-8")
    (OUT_RUN / "summary.json").write_text(
        json.dumps({"best_model": "ngboost", "family_mapping": result.family_mapping,
                    "n_features": len(f38), "n_train": int(len(aug_train))}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # --- Persist per-row predictions + metrics. ---
    for name, frame in (("b0", b0_frame), ("b1", b1), ("b2", b2)):
        out = frame.copy()
        out["residual"] = out["true"] - out["mu"]
        out.to_csv(OUT_RUN / f"loo_{name}.csv", index=False)
    (OUT_RUN / "loo_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== Lab-halide LOO transfer MAE (log10 space) ===")
    print(f"  B0 deployed (lit only)        : MAE={metrics['B0_deployed_lit_only']['mae']:.3f}  cov95={metrics['B0_deployed_lit_only']['coverage95']:.2f}  z_std={metrics['B0_deployed_lit_only']['z_std']:.2f}")
    print(f"  B1 +data, F37                 : MAE={metrics['B1_loo_add_data_f37']['mae']:.3f}")
    print(f"  B2 +data +method, F38         : MAE={metrics['B2_loo_data_plus_method_f38']['mae']:.3f}  cov95={metrics['B2_loo_data_plus_method_f38']['coverage95']:.2f}  z_std={metrics['B2_loo_data_plus_method_f38']['z_std']:.2f}")
    print(f"  Final literature test MAE     : {lit_test_mae:.3f} (baseline v2 ngboost=0.343, n={len(lit_test)})")
    print(f"Output dir: {OUT_RUN.resolve()}")


if __name__ == "__main__":
    main()
