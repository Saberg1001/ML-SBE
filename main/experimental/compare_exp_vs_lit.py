"""Compare experimental measurements against literature values for matching compositions.

Sources compared:
  - data/experimental/experimental-summary.csv  (exp_NNN rows, non-additive)
  - data/experimental/raw/halide.csv additive rows (+5wt%ZrCl4, marked with †)
    matched against the bare host composition's literature value

Literature reference: data/absolute/data-absolute-v3.csv (non-exp rows), best
(highest) conductivity per canonical formula.

Usage:
    python main/experimental/compare_exp_vs_lit.py [--threshold 0.3]
"""

from __future__ import annotations

import argparse
import math
import os
import sys

if __package__ is None:
    _ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)

import numpy as np
import pandas as pd
from pymatgen.core import Composition

from main.experimental.compare_lab_vs_literature import _canonical_formula, _read_halide_lab
from main.paths import DATA_DIR

EXP_SUMMARY = DATA_DIR / "experimental" / "experimental-summary.csv"
V3 = DATA_DIR / "absolute" / "data-absolute-v3.csv"


def _canonical(text: str) -> str | None:
    return _canonical_formula(text)


def _build_lit_index() -> pd.DataFrame:
    """Best (highest) literature conductivity per canonical formula."""
    v3 = pd.read_csv(V3, dtype=str, keep_default_na=False)
    lit = v3[~v3["ID"].str.startswith("exp_", na=False)].copy()
    lit["canon"] = lit["Reduced Composition"].map(_canonical)
    lit["lit_val"] = pd.to_numeric(lit["Ionic conductivity (S cm-1)"], errors="coerce")
    lit = lit.dropna(subset=["canon", "lit_val"])
    lit = lit[lit["lit_val"] > 0]
    return (
        lit.sort_values("lit_val", ascending=False)
        .drop_duplicates("canon")
        .set_index("canon")[["lit_val", "DOI"]]
    )


def main(threshold: float = 0.3) -> None:
    lit_best = _build_lit_index()

    # --- source 1: experimental-summary.csv (non-additive exp rows) ---
    exp = pd.read_csv(EXP_SUMMARY, dtype=str, keep_default_na=False)
    exp["canon"] = exp["化学式"].map(_canonical)
    exp["exp_val"] = pd.to_numeric(exp["电导率（S/cm）"], errors="coerce")

    # --- source 2: additive rows from raw halide.csv ---
    halide_all = _read_halide_lab()
    additive = halide_all[halide_all["additive"]].copy()
    # canonical for additive rows is already the bare host (split at "+")
    additive = additive.rename(columns={"conductivity_S_cm": "exp_val", "formula": "化学式"})

    rows: list[dict] = []

    def _lookup(formula: str, exp_val: float, exp_id: str, additive_flag: bool) -> None:
        canon = _canonical(formula.split("+")[0]) if additive_flag else _canonical(formula)
        if not canon or pd.isna(exp_val) or exp_val <= 0:
            return
        if canon not in lit_best.index:
            return
        lit_row = lit_best.loc[canon]
        lit_val = float(lit_row["lit_val"])
        doi = str(lit_row["DOI"]).strip()
        delta = math.log10(exp_val) - math.log10(lit_val)
        if abs(delta) < threshold:
            return
        rows.append({
            "ID": exp_id,
            "实验成分": formula,
            "实验值": exp_val,
            "文献值": lit_val,
            "Δlog₁₀": round(delta, 2),
            "文献DOI": doi,
            "additive": additive_flag,
        })

    for _, row in exp.iterrows():
        _lookup(row["化学式"], row["exp_val"], row["ID"], False)

    for _, row in additive.iterrows():
        _lookup(row["化学式"], row["exp_val"], row["ID"], True)

    if not rows:
        print(f"没有偏差 > {threshold} 的行。")
        return

    df = pd.DataFrame(rows).sort_values("Δlog₁₀")
    max_abs = df["Δlog₁₀"].abs().max()

    print(f"\n实验 vs 文献对照（|Δlog₁₀| ≥ {threshold}，按 Δlog₁₀ 升序）")
    print(f"  ★ = 偏差最大行   † = 含添加剂行（+5wt%ZrCl4），与裸主体文献值对比")
    print()
    print(f"{'实验成分':<36} {'实验值':>10} {'文献值':>12} {'Δlog₁₀':>8}  文献 DOI")
    print("-" * 108)
    for _, r in df.iterrows():
        star = "★" if abs(r["Δlog₁₀"]) == max_abs else " "
        dag = " †" if r["additive"] else "  "
        formula_col = r["实验成分"] + dag
        exp_str = f"{r['实验值']:.2e}"
        lit_str = f"{r['文献值']:.2e}"
        print(f"{star} {formula_col:<34} {exp_str:>10} {lit_str:>12} {r['Δlog₁₀']:>8.2f}  {r['文献DOI']}")

    n_add = df["additive"].sum()
    n_plain = len(df) - n_add
    print(f"\n共 {len(df)} 条（普通实验行 {n_plain}，含添加剂行 {n_add}）")
    print(f"中位 Δlog₁₀ = {df['Δlog₁₀'].median():.2f}   均值 = {df['Δlog₁₀'].mean():.2f}")
    print(f"全部 exp 行: {len(exp)} 普通 + {len(additive)} 含添加剂，"
          f"其中 {len(df)} 条匹配到文献且偏差 ≥ {threshold}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.3,
                        help="只显示 |Δlog₁₀| 大于此值的行 (默认 0.3)")
    args = parser.parse_args()
    main(args.threshold)
