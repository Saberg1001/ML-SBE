"""
Add non-redundant Small/Kong-style composition features.

Pattern suffixes:
  - all: all elements
  - l: all elements except Li
  - c: cations including Li
  - c_l: non-Li host cations
  - a: anions only
"""

from __future__ import annotations

import argparse
import math
from functools import lru_cache
from pathlib import Path
import warnings

import pandas as pd
import numpy as np
from mendeleev import element as mendeleev_element
from pymatgen.core import Composition, Element

try:
    from get_feature import classify_elements, ionic_radius_pm, oxidation_state_guesses
except ImportError:
    from get_feature.get_feature import classify_elements, ionic_radius_pm, oxidation_state_guesses

warnings.filterwarnings('ignore')


ROOT = Path(__file__).resolve().parents[1]
FEATURE_DIR = ROOT / "features"
DEFAULT_INPUT = FEATURE_DIR / "ionic_26_features_all.csv"
DEFAULT_OUTPUT = FEATURE_DIR / "ionic_34_features_small_all.csv"
BOHR_RADIUS_PM = 52.9177210903

FEATURE_SPECS = [
    ("ir_mean_square_all", "ionic_radius", "mean_square", "all"),
    ("d_mean_c", "d_electrons", "mean", "c"),
    ("r_mean_l", "bond_radius", "mean", "l"),
    ("s_mean_square_c", "s_electrons", "mean_square", "c"),
    ("entropy_c_l", None, "entropy", "c_l"),
    ("Vs_variance_c_l", "s_orbital_volume", "variance", "c_l"),
    ("mp_variance_c_l", "melting_point", "variance", "c_l"),
    ("atwt_geometric_mean_all", "atomic_mass", "geometric_mean", "all"),
]


# ====================================================================
# Element property lookup
# ====================================================================

@lru_cache(maxsize=None)
def s_orbital_radius_pm(symbol):
    """Estimate valence s orbital radius from Clementi effective charge."""
    el = mendeleev_element(symbol)
    s_shells = [item.n for item in el.screening_constants if item.s == "s"]
    if not s_shells:
        return 0
    n = max(s_shells)
    zeff = el.zeff(n=n, o="s", method="clementi")
    if zeff is None or zeff == 0:
        zeff = el.zeff(n=n, o="s", method="slater")
    if zeff is None or zeff == 0:
        return 0
    return BOHR_RADIUS_PM * n**2 / zeff


@lru_cache(maxsize=None)
def bond_radius_pm(symbol):
    """Covalent radius for nonmetals and metallic radius for metals."""
    pymatgen_element = Element(symbol)
    mendeleev_el = mendeleev_element(symbol)
    if pymatgen_element.is_metal and mendeleev_el.metallic_radius is not None:
        return float(mendeleev_el.metallic_radius)
    if mendeleev_el.covalent_radius is not None:
        return float(mendeleev_el.covalent_radius)
    return pymatgen_element.atomic_radius * 100 if pymatgen_element.atomic_radius else 0


@lru_cache(maxsize=None)
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
        elif property_name == 'bond_radius':
            return bond_radius_pm(element)
        elif property_name == 'ionic_radius':
            ox = el.common_oxidation_states[0] if el.common_oxidation_states else 0
            return el.ionic_radii.get(ox, 0) * 100  # pm
        elif property_name == 'atomic_mass':
            return float(el.atomic_mass)
        elif property_name == 'atomic_number':
            return float(el.Z)
        elif property_name == 'mendeleev_no':
            value = el.data.get("Mendeleev no")
            return float(value) if value is not None else 0
        elif property_name == 'thermal_conductivity':
            value = el.data.get("Thermal conductivity")
            return float(value) if value is not None else 0
        elif property_name == 'density_solid':
            value = el.data.get("Density of solid")
            return float(value) if value is not None else 0
        elif property_name == 'first_ionization_energy':
            values = el.data.get("Ionization energies") or []
            return float(values[0]) if values else 0
        elif property_name == 's_orbital_volume':
            radius_pm = s_orbital_radius_pm(element)
            return math.pi * radius_pm**3
        elif property_name == 'melting_point':
            value = el.data.get("Melting point")
            return float(value) if value is not None else 0
        else:
            return 0
    except:
        return 0


# ====================================================================
# Statistical aggregation
# ====================================================================

def composition_groups(comp):
    """Return element groups for Small composition patterns."""
    elements = [el.symbol for el in comp.elements]
    oxidation_guesses, _, _, _ = oxidation_state_guesses(comp)
    oxidation_guess = oxidation_guesses[0] if oxidation_guesses else {}
    classification = classify_elements(elements, oxidation_guess)
    return {
        "all": set(elements),
        "l": {symbol for symbol in elements if symbol != "Li"},
        "c": {
            symbol for symbol in elements
            if classification.get(symbol) in ("Li", "host_cation")
        },
        "c_l": {
            symbol for symbol in elements
            if classification.get(symbol) == "host_cation"
        },
        "a": {
            symbol for symbol in elements
            if classification.get(symbol) in ("anion", "halide")
        },
    }


def weighted_values(comp, property_name, pattern):
    """Collect property values and composition weights for one pattern."""
    allowed = composition_groups(comp)[pattern]
    oxidation_guesses, _, _, _ = oxidation_state_guesses(comp)
    oxidation_guess = oxidation_guesses[0] if oxidation_guesses else {}
    values, weights = [], []
    for el, amt in comp.items():
        if el.symbol in allowed:
            if property_name == "ionic_radius":
                charge = oxidation_guess.get(el.symbol)
                value = ionic_radius_pm(el.symbol, charge) if charge is not None else None
                if value is None:
                    value = get_element_property(el.symbol, property_name)
            else:
                value = get_element_property(el.symbol, property_name)
            values.append(value)
            weights.append(amt)
    return values, weights


def compute_weighted_stat(comp, property_name, statistic, pattern):
    """Compute one weighted statistic for a Small-style feature."""
    values, weights = weighted_values(comp, property_name, pattern)
    if not values:
        return 0.0

    values = np.array(values, dtype=float)
    weights = np.array(weights, dtype=float)
    weights /= weights.sum()

    if statistic == "mean":
        return float(np.average(values, weights=weights))
    if statistic == "mean_square":
        return float(np.average(values ** 2, weights=weights))
    if statistic == "variance":
        mean = np.average(values, weights=weights)
        return float(np.average((values - mean) ** 2, weights=weights))
    if statistic == "geometric_mean":
        if np.any(values <= 0):
            return 0.0
        return float(np.exp(np.average(np.log(values), weights=weights)))
    raise ValueError(f"Unsupported statistic: {statistic}")


def compute_entropy(comp, pattern):
    """Configurational entropy for one Small composition pattern."""
    allowed = composition_groups(comp)[pattern]
    amounts = {
        el.symbol: amt for el, amt in comp.items()
        if el.symbol in allowed
    }
    if len(amounts) <= 1:
        return 0.0
    total = sum(amounts.values())
    return -sum((a / total) * np.log(a / total) for a in amounts.values() if a > 0)


def compute_count(comp, pattern):
    """Count distinct elements in one Small composition pattern."""
    return float(len(composition_groups(comp)[pattern]))


# ====================================================================
# Main feature extraction
# ====================================================================

def add_small_top10_features(df):
    """
    Add 10 non-redundant features inspired by Small (Kong 2025).

    The feature scopes follow the requested pattern mapping:
      - mean of d and mean square of s: c
      - mean of r: l
      - variance of Vs = pi * rs^3 and mp: c_l
      - remaining selected features: all
    """

    print("Parsing compositions...")
    compositions = df['True Composition'].apply(
        lambda x: Composition(x) if pd.notna(x) else None)
    valid = compositions.notna().sum()
    print(f"  Valid: {valid}/{len(df)}")

    print(f"\nComputing {len(FEATURE_SPECS)} new features...")
    new_cols = []
    total_new_features = len(FEATURE_SPECS)
    for index, (column, property_name, statistic, pattern) in enumerate(FEATURE_SPECS, start=1):
        print(f" {index:2d}/{total_new_features}  {column}")
        if statistic == "entropy":
            df[column] = compositions.apply(
                lambda c: compute_entropy(c, pattern) if c else 0.0)
        elif statistic == "count":
            df[column] = compositions.apply(
                lambda c: compute_count(c, pattern) if c else 0.0)
        else:
            df[column] = compositions.apply(
                lambda c: compute_weighted_stat(c, property_name, statistic, pattern) if c else 0.0)
        new_cols.append(column)

    print(f"\n+ Added {len(new_cols)} features -> total columns: {len(df.columns)}")
    return df, new_cols


# ====================================================================
# Entry point
# ====================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add non-redundant Small-paper-inspired features to a base feature table."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("Small Paper Feature Extraction (non-redundant top features)")
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
