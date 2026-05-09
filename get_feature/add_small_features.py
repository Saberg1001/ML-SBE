"""
Add Top 10 non-redundant features from Small (Kong 2025) paper.

Excluded (redundant with existing features):
  - Li_content_ratio ↔ n_Li (r=1.00)
  - O_content_ratio ↔ r⁻/χ⁻/ρ⁻ (r>0.92)
  - covalent_radius_std_nonLi ↔ r⁺(excl Li⁺) - r⁻ (r=0.976)

Replaced with independent alternatives:
  - atomic_mass_mean_nonLi (原子量均值)
  - ir_std_nonLi (离子半径标准差)
  - n_elements (元素数量 → 组成复杂度代理)
"""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import pandas as pd
import numpy as np
from pymatgen.core import Composition, Element
warnings.filterwarnings('ignore')


ROOT = Path(__file__).resolve().parents[1]
FEATURE_DIR = ROOT / "features"
DEFAULT_INPUT = FEATURE_DIR / "ionic_26_features_all.csv"
DEFAULT_OUTPUT = FEATURE_DIR / "ionic_36_features_small_all.csv"


# ====================================================================
# Element property lookup
# ====================================================================

def get_element_property(element, property_name):
    """Get element property with fallback to 0."""
    try:
        el = Element(element)
        if property_name == 'd_electrons':
            return sum(c for (_, orb, c) in el.full_electronic_structure if orb == 'd')
        elif property_name == 's_electrons':
            return sum(c for (_, orb, c) in el.full_electronic_structure if orb == 's')
        elif property_name == 'p_electrons':
            return sum(c for (_, orb, c) in el.full_electronic_structure if orb == 'p')
        elif property_name == 'covalent_radius':
            return el.atomic_radius * 100 if el.atomic_radius else 0  # pm
        elif property_name == 'ionic_radius':
            ox = el.common_oxidation_states[0] if el.common_oxidation_states else 0
            return el.ionic_radii.get(ox, 0) * 100  # pm
        elif property_name == 'atomic_mass':
            return float(el.atomic_mass)
        else:
            return 0
    except:
        return 0


# ====================================================================
# Statistical aggregation
# ====================================================================

def compute_weighted_stats(comp, property_name, exclude_elements=None):
    """
    Compute weighted mean/std/min/max/range for an elemental property
    over the composition, excluding specified elements.
    """
    if exclude_elements is None:
        exclude_elements = []

    values, weights = [], []
    for el, amt in comp.items():
        if el.symbol not in exclude_elements:
            values.append(get_element_property(el.symbol, property_name))
            weights.append(amt)

    if not values:
        return {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'range': 0}

    values = np.array(values, dtype=float)
    weights = np.array(weights, dtype=float)
    weights /= weights.sum()

    mean = np.average(values, weights=weights)
    std = np.sqrt(np.average((values - mean) ** 2, weights=weights))
    return {'mean': mean, 'std': std, 'min': values.min(),
            'max': values.max(), 'range': values.max() - values.min()}


def compute_cation_entropy(comp):
    """Configurational entropy of non-Li cations: -Sigma p_i ln(p_i)."""
    cations = {el.symbol: amt for el, amt in comp.items()
                if el.is_metal and el.symbol != 'Li'}
    if len(cations) <= 1:
        return 0.0
    total = sum(cations.values())
    return -sum((a / total) * np.log(a / total) for a in cations.values() if a > 0)


# ====================================================================
# Main feature extraction
# ====================================================================

def add_small_top10_features(df):
    """
    Add 10 non-redundant features inspired by Small (Kong 2025) paper.

    New features (10):
      1. ir_mean_nonLi          - Effective ionic radius mean (non-Li)
      2. d_electrons_mean_nonLi - d-orbital electrons mean (non-Li)
      3. covalent_radius_mean_nonLi - Covalent radius mean (non-Li)
      4. cation_entropy         - Configurational entropy of non-Li cations
      5. d_electrons_std_nonLi  - d-orbital electrons std (non-Li)
      6. s_electrons_mean_nonLi - s-orbital electrons mean (non-Li)
      7. p_electrons_mean_nonLi - p-orbital electrons mean (non-Li)
      8. atomic_mass_mean_nonLi - Atomic mass mean (non-Li)
      9. ir_std_nonLi           - Effective ionic radius std (non-Li)
     10. n_elements             - Number of distinct elements

    NOT added (redundant with existing features):
      - Li_content_ratio       <-> n_Li (r=1.00)
      - O_content_ratio        <-> r-/chi-/rho- (r>0.92)
      - covalent_radius_std    <-> r+(excl Li+)-r- (r=0.976)
    """

    print("Parsing compositions...")
    compositions = df['True Composition'].apply(
        lambda x: Composition(x) if pd.notna(x) else None)
    valid = compositions.notna().sum()
    print(f"  Valid: {valid}/{len(df)}")

    print("\nComputing 10 new features...")

    # 1. Effective ionic radius mean (non-Li)
    print("  1/10  ir_mean_nonLi")
    ir_stats = compositions.apply(
        lambda c: compute_weighted_stats(c, 'ionic_radius', ['Li']) if c else {})
    df['ir_mean_nonLi'] = ir_stats.apply(lambda x: x.get('mean', 0))

    # 2. d-electrons mean (non-Li)
    print("  2/10  d_electrons_mean_nonLi")
    d_stats = compositions.apply(
        lambda c: compute_weighted_stats(c, 'd_electrons', ['Li']) if c else {})
    df['d_electrons_mean_nonLi'] = d_stats.apply(lambda x: x.get('mean', 0))

    # 3. Covalent radius mean (non-Li)
    print("  3/10  covalent_radius_mean_nonLi")
    cr_stats = compositions.apply(
        lambda c: compute_weighted_stats(c, 'covalent_radius', ['Li']) if c else {})
    df['covalent_radius_mean_nonLi'] = cr_stats.apply(lambda x: x.get('mean', 0))

    # 4. Cation entropy
    print("  4/10  cation_entropy")
    df['cation_entropy'] = compositions.apply(
        lambda c: compute_cation_entropy(c) if c else 0)

    # 5. d-electrons std (non-Li)
    print("  5/10  d_electrons_std_nonLi")
    df['d_electrons_std_nonLi'] = d_stats.apply(lambda x: x.get('std', 0))

    # 6. s-electrons mean (non-Li)
    print("  6/10  s_electrons_mean_nonLi")
    s_stats = compositions.apply(
        lambda c: compute_weighted_stats(c, 's_electrons', ['Li']) if c else {})
    df['s_electrons_mean_nonLi'] = s_stats.apply(lambda x: x.get('mean', 0))

    # 7. p-electrons mean (non-Li)
    print("  7/10  p_electrons_mean_nonLi")
    p_stats = compositions.apply(
        lambda c: compute_weighted_stats(c, 'p_electrons', ['Li']) if c else {})
    df['p_electrons_mean_nonLi'] = p_stats.apply(lambda x: x.get('mean', 0))

    # 8. Atomic mass mean (non-Li)  [replaces redundant Li_content_ratio]
    print("  8/10  atomic_mass_mean_nonLi")
    am_stats = compositions.apply(
        lambda c: compute_weighted_stats(c, 'atomic_mass', ['Li']) if c else {})
    df['atomic_mass_mean_nonLi'] = am_stats.apply(lambda x: x.get('mean', 0))

    # 9. Ionic radius std (non-Li)  [replaces redundant O_content_ratio]
    print("  9/10  ir_std_nonLi")
    df['ir_std_nonLi'] = ir_stats.apply(lambda x: x.get('std', 0))

    # 10. Number of distinct elements  [replaces redundant covalent_radius_std]
    print(" 10/10  n_elements")
    df['n_elements'] = compositions.apply(
        lambda c: len(c) if c else 0)

    new_cols = [
        'ir_mean_nonLi', 'd_electrons_mean_nonLi', 'covalent_radius_mean_nonLi',
        'cation_entropy', 'd_electrons_std_nonLi', 's_electrons_mean_nonLi',
        'p_electrons_mean_nonLi', 'atomic_mass_mean_nonLi', 'ir_std_nonLi',
        'n_elements'
    ]
    print(f"\n+ Added {len(new_cols)} features -> total columns: {len(df.columns)}")
    return df, new_cols


# ====================================================================
# Entry point
# ====================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add 10 Small-paper-inspired features to the base 26-feature table."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("Small Paper Feature Extraction (non-redundant Top 10)")
    print("=" * 65)

    df = pd.read_csv(args.input)
    print(f"\nInput:  {args.input}")
    print(f"Shape:  {df.shape[0]} rows x {df.shape[1]} cols")

    df_out, new_cols = add_small_top10_features(df)

    df_out.to_csv(args.output, index=False)
    print(f"\nOutput: {args.output}")
    print(f"Shape:  {df_out.shape[0]} rows x {df_out.shape[1]} cols")

    # ---- Quick validation ----
    print("\n" + "=" * 65)
    print("New feature summary")
    print("=" * 65)
    print(df_out[new_cols].describe().round(3).to_string())

    # Correlation with target
    def parse_cond(v):
        if pd.isna(v): return np.nan
        if isinstance(v, (int, float)): return float(v)
        s = str(v).strip().replace('\u2212', '-')
        if s.startswith('<'): return 1e-11
        try: return float(s)
        except: return np.nan

    df_out['_log_cond'] = df_out['Ionic conductivity (S cm-1)'].apply(
        lambda v: np.log10(parse_cond(v)))

    print("\n" + "=" * 65)
    print("Correlation with log10(conductivity)")
    print("=" * 65)
    corrs = df_out[new_cols + ['_log_cond']].corr()['_log_cond'].drop('_log_cond')
    corrs = corrs.reindex(corrs.abs().sort_values(ascending=False).index)
    for feat, r in corrs.items():
        print(f"  {feat:<30s}  r = {r:+.4f}")

    # Check redundancy with original features
    metadata = ['ID', 'True Composition', 'Z_by_element',
                'Ionic conductivity (S cm-1)', '_log_cond']
    old_cols = [c for c in df_out.columns if c not in metadata + new_cols]

    print("\n" + "=" * 65)
    print("Max |correlation| with existing features (should be < 0.9)")
    print("=" * 65)
    for nf in new_cols:
        max_corr = 0
        max_feat = ''
        for of in old_cols:
            r = df_out[[nf, of]].apply(pd.to_numeric, errors='coerce').corr().iloc[0, 1]
            if abs(r) > abs(max_corr):
                max_corr = r
                max_feat = of
        status = "OK" if abs(max_corr) < 0.9 else "WARNING"
        print(f"  {nf:<30s}  max|r|={abs(max_corr):.3f}  ({max_feat})  [{status}]")

    df_out.drop(columns=['_log_cond'], inplace=True)
    df_out.to_csv(args.output, index=False)

    print("\n" + "=" * 65)
    print("DONE")
    print("=" * 65)
