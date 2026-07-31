from __future__ import annotations

from dataclasses import asdict, dataclass, field
from functools import lru_cache
from itertools import product
import json
import math
from pathlib import Path
import re

from mendeleev import element
import numpy as np
import pandas as pd
from pymatgen.core import Composition
from pymatgen.core.periodic_table import Element

from .paths import CHEMISTRY_CONFIG_DIR, MODELING_DIR


DEFAULT_FEATURE_PATH = MODELING_DIR / "absolute" / "generated_features.csv"
OXIDATION_CONFIG = CHEMISTRY_CONFIG_DIR / "oxidation_states.json"
IONIC_RADIUS_CONFIG = CHEMISTRY_CONFIG_DIR / "ionic_radius_overrides.json"

TARGET_COLUMN = "Ionic conductivity (S cm-1)"
FAMILY_COLUMN = "Family"
FAMILY_FEATURE_COLUMN = "family"
CHARGE_RESIDUAL_LIMIT = 1.0
CHARGE_RESIDUAL_EXCLUDE_LIMIT = 6.0
MANUAL_ABNORMAL_CHARGE_IDS = frozenset({
    "mtr",
    "s2v",
    "5z8",
    "ueu",
    "uwh",
    "a6v",
})
ORGANIC_MARKER_ELEMENTS = frozenset({"C", "H"})
ORGANIC_NEUTRAL_ELEMENTS = frozenset({"C", "H", "N", "O"})
HALIDE_ELEMENTS = frozenset({"F", "Cl", "Br", "I"})
ANION_ELEMENTS = frozenset({"O", "S", "Se", "Te", "N", "P", "F", "Cl", "Br", "I"})
ELEMENTARY_CHARGE_C = 1.602e-19
PM_TO_M = 1e-12
BOHR_RADIUS_PM = 52.9177210903

SMALL_FEATURE_SPECS = (
    ("ir_mean_square_all", "ionic_radius", "mean_square", "all"),
    ("d_mean_c", "d_electrons", "mean", "c"),
    ("r_mean_l", "bond_radius", "mean", "l"),
    ("s_mean_square_c", "s_electrons", "mean_square", "c"),
    ("entropy_c_l", None, "entropy", "c_l"),
    ("Vs_variance_c_l", "s_orbital_volume", "variance", "c_l"),
    ("mp_variance_c_l", "melting_point", "variance", "c_l"),
    ("atwt_geometric_mean_all", "atomic_mass", "geometric_mean", "all"),
)

METADATA_COLUMNS = {
    "ID",
    "Reduced Composition",
    "True Composition",
    FAMILY_COLUMN,
    TARGET_COLUMN,
    "IC (Total)",
    "IC (Bulk)",
    "Space group",
    "Space group #",
    "Z",
    "a",
    "b",
    "c",
    "alpha",
    "beta",
    "gamma",
    "DOI",
    "Checked",
    "Ref",
    "Cif ID",
    "Cif ref_1",
    "Cif ref_2",
    "note",
    "close match",
    "close match DOI",
    "ICSD ID",
    "Laskowski ID",
    "Liion ID",
    "conductivity_value", #数值
    "conductivity_used", #真实训练值
    "conductivity_qualifier", #数值类型，比如不高于，不低于，空值等
    "log10_conductivity",
    "sample_weight",
}


@lru_cache(maxsize=None)
def pauling_electronegativity(symbol: str) -> float | None:
    value = element(symbol).electronegativity("pauling")
    return None if value is None else float(value)


@lru_cache(maxsize=None)
def oxidation_state_config() -> dict[str, tuple[float, ...]]:
    with OXIDATION_CONFIG.open("r", encoding="utf-8") as file:
        data = json.load(file)
    return {
        symbol: tuple(float(state) for state in states)
        for symbol, states in data["oxidation_states"].items()
    }


@lru_cache(maxsize=None)
def candidate_oxidation_states(symbol: str) -> tuple[float, ...]:
    config_states = oxidation_state_config().get(symbol)
    if config_states:
        return config_states
    elem = Element(symbol)
    states = elem.common_oxidation_states or elem.oxidation_states
    return tuple(float(state) for state in states) or (0.0,)


def contains_organic_molecule(composition: Composition) -> bool:
    symbols = {elem.symbol for elem in composition.elements}
    return ORGANIC_MARKER_ELEMENTS.issubset(symbols)


def neutral_organic_elements(composition: Composition) -> set[str]:
    if not contains_organic_molecule(composition):
        return set()
    return {
        elem.symbol
        for elem in composition.elements
        if elem.symbol in ORGANIC_NEUTRAL_ELEMENTS
    }


def oxidation_state_guesses(
    composition: Composition,
) -> tuple[list[dict[str, float]], str, str, str]:
    integer_formula, _ = composition.get_integer_formula_and_factor()
    amounts = composition.get_el_amt_dict()
    ignored_symbols = neutral_organic_elements(composition)
    fixed_guess: dict[str, float] = {}
    variable_symbols: list[str] = []
    variable_candidates: list[tuple[float, ...]] = []
    for elem in composition.elements:
        symbol = elem.symbol
        if symbol in ignored_symbols:
            continue
        states = oxidation_state_config().get(symbol)
        if not states:
            states = candidate_oxidation_states(symbol)
        if len(states) == 1:
            fixed_guess[symbol] = states[0]
        else:
            variable_symbols.append(symbol)
            variable_candidates.append(states)
    if not variable_symbols:
        residual_charge = sum(
            amounts[symbol] * charge
            for symbol, charge in fixed_guess.items()
        )
        note = ""
        if ignored_symbols:
            note = "neutral organic elements ignored: " + ",".join(sorted(ignored_symbols))
        if abs(residual_charge) >= 1e-8:
            residual_note = f"residual_charge={residual_charge:.6g}"
            note = f"{note}; {residual_note}" if note else residual_note
        return [fixed_guess], integer_formula, note, "config_charge_balance"

    best_guess = None
    best_score = float("inf")
    for states in product(*variable_candidates):
        guess = fixed_guess | {
            symbol: float(state)
            for symbol, state in zip(variable_symbols, states)
        }
        residual_charge = sum(
            amounts[symbol] * charge
            for symbol, charge in guess.items()
        )
        exact_penalty = abs(residual_charge)
        state_penalty = 0.001 * sum(abs(float(state)) for state in states)
        score = exact_penalty + state_penalty
        if score < best_score:
            best_score = score
            best_guess = guess
    if best_guess is None:
        return [], integer_formula, "config charge-balance search failed", "config_charge_balance"

    residual_charge = sum(
        amounts[symbol] * charge
        for symbol, charge in best_guess.items()
    )
    note = ""
    if ignored_symbols:
        note = "neutral organic elements ignored: " + ",".join(sorted(ignored_symbols))
    if abs(residual_charge) >= 1e-8:
        residual_note = f"residual_charge={residual_charge:.6g}"
        note = f"{note}; {residual_note}" if note else residual_note
    return [best_guess], integer_formula, note, "config_charge_balance"


def charge_residual(
    amounts: dict[str, float],
    oxidation_guess: dict[str, float],
) -> float:
    return sum(
        amounts[symbol] * charge
        for symbol, charge in oxidation_guess.items()
    )


def element_value_list(values: dict[str, object]) -> str:
    items = [
        {"element": symbol, "value": value}
        for symbol, value in values.items()
    ]
    return json.dumps(items, ensure_ascii=False)


def charge_balance_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for _, row in frame.iterrows():
        formula = row["True Composition"]
        composition = Composition(formula)
        amounts = {
            symbol: float(composition.get_el_amt_dict()[symbol])
            for symbol in (elem.symbol for elem in composition.elements)
        }
        oxidation_guesses, _, _, _ = oxidation_state_guesses(composition)
        oxidation_guess = oxidation_guesses[0] if oxidation_guesses else {}
        residual = charge_residual(amounts, oxidation_guess)
        records.append(
            {
                "ID": row["ID"],
                "True Composition": formula,
                TARGET_COLUMN: row[TARGET_COLUMN],
                "residual_charge": residual,
                "oxidation_guess": element_value_list(oxidation_guess),
            }
        )
    return records


@lru_cache(maxsize=None)
def ionic_radius_overrides() -> dict[tuple[str, int], dict[str, object]]:
    if not IONIC_RADIUS_CONFIG.exists():
        return {}
    with IONIC_RADIUS_CONFIG.open("r", encoding="utf-8") as file:
        data = json.load(file)
    overrides: dict[tuple[str, int], dict[str, object]] = {}
    for item in data.get("ionic_radius_overrides_pm", []):
        symbol = str(item["element"])
        charge = int(item["charge"])
        overrides[(symbol, charge)] = item
    return overrides


def mendeleev_ionic_radius_pm(symbol: str, charge_int: int) -> float | None:
    radii = [
        r
        for r in element(symbol).ionic_radii
        if r.charge == charge_int and r.ionic_radius is not None
    ]
    if not radii:
        return None
    reliable = [
        r.ionic_radius
        for r in radii
        if getattr(r, "most_reliable", False)
    ]
    values = [float(v) for v in reliable] if reliable else [float(r.ionic_radius) for r in radii]
    return float(sum(values) / len(values))


@lru_cache(maxsize=None)
def ionic_radius_pm(symbol: str, charge: float) -> float | None:
    if charge == 0 or not float(charge).is_integer():
        return None
    charge_int = int(charge)
    override = ionic_radius_overrides().get((symbol, charge_int))
    if override is not None:
        if "radius_pm" in override:
            return float(override["radius_pm"])
        if "fallback_charge" in override:
            return mendeleev_ionic_radius_pm(symbol, int(override["fallback_charge"]))
    return mendeleev_ionic_radius_pm(symbol, charge_int)


def safe_weighted_average(
    symbols: list[str],
    amounts: dict[str, float],
    values: dict[str, float | None],
) -> float | None:
    valid_items = [
        (amounts[s], values[s])
        for s in symbols
        if values.get(s) is not None
    ]
    if not valid_items:
        return None
    total = sum(a for a, _ in valid_items)
    if total == 0:
        return None
    return sum(a * v for a, v in valid_items) / total


def safe_diff(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return a - b


def safe_ratio(a: float | None, b: float | None) -> float | None:
    if a is None or b is None or b == 0:
        return None
    return a / b


def charge_density_c_m3(charge: float, radius_pm: float | None) -> float | None:
    if radius_pm is None:
        return None
    if radius_pm == 0:
        return 0.0
    radius_m = radius_pm * PM_TO_M
    return 3 * abs(charge) * ELEMENTARY_CHARGE_C / (4 * math.pi * radius_m**3)


def ionic_potential(charge: float, radius_pm: float | None) -> float | None:
    if radius_pm is None or radius_pm == 0:
        return None
    return abs(charge) / radius_pm


def classify_elements(
    elements: list[str],
    oxidation_guess: dict[str, float],
) -> dict[str, str]:
    classification: dict[str, str] = {}
    for symbol in elements:
        charge = oxidation_guess.get(symbol)
        if symbol == "Li":
            classification[symbol] = "Li"
        elif charge is not None and symbol == "H" and charge > 0:
            classification[symbol] = "proton"
        elif charge is not None and charge < 0:
            classification[symbol] = "halide" if symbol in HALIDE_ELEMENTS else "anion"
        elif charge is not None and charge > 0:
            classification[symbol] = "host_cation"
        elif symbol in HALIDE_ELEMENTS:
            classification[symbol] = "halide"
        elif symbol in ANION_ELEMENTS:
            classification[symbol] = "anion"
        else:
            classification[symbol] = "host_cation"
    return classification


def composition_features(formula: str) -> pd.Series:
    composition = Composition(formula)
    elements = [elem.symbol for elem in composition.elements]
    amounts = {
        symbol: float(composition.get_el_amt_dict()[symbol])
        for symbol in elements
    }
    total_atoms = sum(amounts.values())

    oxidation_guesses, _, _, _ = oxidation_state_guesses(composition)
    ox = oxidation_guesses[0] if oxidation_guesses else {}

    en_map: dict[str, float | None] = {}
    r_map: dict[str, float | None] = {}
    rho_map: dict[str, float | None] = {}
    phi_map: dict[str, float | None] = {}
    for symbol in elements:
        en_map[symbol] = pauling_electronegativity(symbol)
        charge = ox.get(symbol)
        if charge is not None:
            r_map[symbol] = ionic_radius_pm(symbol, charge)
            rho_map[symbol] = charge_density_c_m3(charge, r_map[symbol])
            phi_map[symbol] = ionic_potential(charge, r_map[symbol])
        else:
            r_map[symbol] = None
            rho_map[symbol] = None
            phi_map[symbol] = None

    cls = classify_elements(elements, ox)
    anion_sym = [s for s in elements if cls.get(s) in ("anion", "halide")]
    cat_incl_sym = [s for s in elements if cls.get(s) in ("Li", "host_cation")]
    cat_excl_sym = [s for s in elements if cls.get(s) == "host_cation"]
    halide_sym = [s for s in elements if cls.get(s) == "halide"]
    non_halide_anion_sym = [s for s in elements if cls.get(s) == "anion"]

    chi_all = safe_weighted_average(elements, amounts, en_map)
    chi_minus = safe_weighted_average(anion_sym, amounts, en_map)
    chi_plus_incl = safe_weighted_average(cat_incl_sym, amounts, en_map)
    chi_plus_excl = safe_weighted_average(cat_excl_sym, amounts, en_map)

    r_all = safe_weighted_average(elements, amounts, r_map)
    r_minus = safe_weighted_average(anion_sym, amounts, r_map)
    r_plus_incl = safe_weighted_average(cat_incl_sym, amounts, r_map)
    r_plus_excl = safe_weighted_average(cat_excl_sym, amounts, r_map)

    rho_all = safe_weighted_average(elements, amounts, rho_map)
    rho_minus = safe_weighted_average(anion_sym, amounts, rho_map)
    rho_plus_incl = safe_weighted_average(cat_incl_sym, amounts, rho_map)
    rho_plus_excl = safe_weighted_average(cat_excl_sym, amounts, rho_map)

    phi_plus_incl = safe_weighted_average(cat_incl_sym, amounts, phi_map)
    phi_plus_excl = safe_weighted_average(cat_excl_sym, amounts, phi_map)

    en_values = [value for value in en_map.values() if value is not None]
    chi_max_min = (max(en_values) - min(en_values)) if len(en_values) >= 2 else None

    return pd.Series(
        {
            "Z_by_element": element_value_list(ox),
            "χₐₗₗ": chi_all,
            "χ⁻": chi_minus,
            "χ⁺(incl Li⁺)": chi_plus_incl,
            "χ⁺(excl Li⁺)": chi_plus_excl,
            "χ⁺(incl Li⁺) - χ⁻": safe_diff(chi_plus_incl, chi_minus),
            "χ⁺(excl Li⁺) - χ⁻": safe_diff(chi_plus_excl, chi_minus),
            "χₘₐₓ - χₘᵢₙ": chi_max_min,
            "rₐₗₗ (pm)": r_all,
            "r⁻ (pm)": r_minus,
            "r⁺(incl Li⁺) (pm)": r_plus_incl,
            "r⁺(excl Li⁺) (pm)": r_plus_excl,
            "r⁺(excl Li⁺) - r⁻": safe_diff(r_plus_excl, r_minus),
            "r⁺(incl Li⁺) - r⁻": safe_diff(r_plus_incl, r_minus),
            "r⁺(excl Li⁺) / r⁻": safe_ratio(r_plus_excl, r_minus),
            "r⁺(incl Li⁺) / r⁻": safe_ratio(r_plus_incl, r_minus),
            "ρₐₗₗ (C m⁻³)": rho_all,
            "ρ⁻ (C m⁻³)": rho_minus,
            "ρ⁺(incl Li⁺) (C m⁻³)": rho_plus_incl,
            "ρ⁺(excl Li⁺) (C m⁻³)": rho_plus_excl,
            "ρ⁺(incl Li⁺) / ρ⁻": safe_ratio(rho_plus_incl, rho_minus),
            "ρ⁺(incl Li⁺) - ρ⁻": safe_diff(rho_plus_incl, rho_minus),
            "Φ⁺(incl Li⁺) (|Z| pm⁻¹)": phi_plus_incl,
            "Φ⁺(excl Li⁺) (|Z| pm⁻¹)": phi_plus_excl,
            "nₕₐₗᵢdₑ": (
                sum(amounts.get(s, 0) for s in halide_sym) / total_atoms
                if total_atoms > 0 else None
            ),
            "nₕₒₛₜ cₐₜᵢₒₙ": (
                sum(amounts.get(s, 0) for s in cat_excl_sym) / total_atoms
                if total_atoms > 0 else None
            ),
            "nₐₙᵢₒₙ": (
                sum(amounts.get(s, 0) for s in non_halide_anion_sym) / total_atoms
                if total_atoms > 0 else None
            ),
            "n_Li": (
                amounts.get("Li", 0) / total_atoms
                if total_atoms > 0 else None
            ),
        }
    )


@lru_cache(maxsize=None)
def _small_s_orbital_radius_pm(symbol: str) -> float:
    """Estimate the valence-s orbital radius using Clementi screening."""
    elem = element(symbol)
    shells = [item.n for item in elem.screening_constants if item.s == "s"]
    if not shells:
        return 0.0
    shell = max(shells)
    effective_charge = elem.zeff(n=shell, o="s", method="clementi")
    if effective_charge in (None, 0):
        effective_charge = elem.zeff(n=shell, o="s", method="slater")
    if effective_charge in (None, 0):
        return 0.0
    return float(BOHR_RADIUS_PM * shell**2 / effective_charge)


@lru_cache(maxsize=None)
def _small_bond_radius_pm(symbol: str) -> float:
    """Use metallic radius for metals and covalent radius otherwise."""
    pymatgen_element = Element(symbol)
    mendeleev_element = element(symbol)
    if pymatgen_element.is_metal and mendeleev_element.metallic_radius is not None:
        return float(mendeleev_element.metallic_radius)
    if mendeleev_element.covalent_radius is not None:
        return float(mendeleev_element.covalent_radius)
    if pymatgen_element.atomic_radius is not None:
        return float(pymatgen_element.atomic_radius) * 100.0
    return 0.0


@lru_cache(maxsize=None)
def _small_element_property(symbol: str, property_name: str) -> float:
    """Read one elemental property with the historical zero fallback."""
    try:
        elem = Element(symbol)
        if property_name == "d_electrons":
            return float(sum(count for _, orbital, count in elem.full_electronic_structure if orbital == "d"))
        if property_name == "s_electrons":
            return float(sum(count for _, orbital, count in elem.full_electronic_structure if orbital == "s"))
        if property_name == "bond_radius":
            return _small_bond_radius_pm(symbol)
        if property_name == "ionic_radius":
            state = elem.common_oxidation_states[0] if elem.common_oxidation_states else 0
            return float(elem.ionic_radii.get(state, 0)) * 100.0
        if property_name == "atomic_mass":
            return float(elem.atomic_mass)
        if property_name == "s_orbital_volume":
            return float(math.pi * _small_s_orbital_radius_pm(symbol) ** 3)
        if property_name == "melting_point":
            value = elem.data.get("Melting point")
            return float(value) if value is not None else 0.0
    except Exception:
        return 0.0
    return 0.0


def _small_composition_groups(composition: Composition) -> dict[str, set[str]]:
    symbols = [elem.symbol for elem in composition.elements]
    guesses, _, _, _ = oxidation_state_guesses(composition)
    classification = classify_elements(symbols, guesses[0] if guesses else {})
    return {
        "all": set(symbols),
        "l": {symbol for symbol in symbols if symbol != "Li"},
        "c": {symbol for symbol in symbols if classification.get(symbol) in {"Li", "host_cation"}},
        "c_l": {symbol for symbol in symbols if classification.get(symbol) == "host_cation"},
        "a": {symbol for symbol in symbols if classification.get(symbol) in {"anion", "halide"}},
    }


def _small_weighted_values(
    composition: Composition,
    property_name: str,
    pattern: str,
) -> tuple[np.ndarray, np.ndarray]:
    allowed = _small_composition_groups(composition)[pattern]
    guesses, _, _, _ = oxidation_state_guesses(composition)
    oxidation_guess = guesses[0] if guesses else {}
    values: list[float] = []
    weights: list[float] = []
    for elem, amount in composition.items():
        if elem.symbol not in allowed:
            continue
        if property_name == "ionic_radius":
            charge = oxidation_guess.get(elem.symbol)
            value = ionic_radius_pm(elem.symbol, charge) if charge is not None else None
            if value is None:
                value = _small_element_property(elem.symbol, property_name)
        else:
            value = _small_element_property(elem.symbol, property_name)
        values.append(float(value))
        weights.append(float(amount))
    return np.asarray(values, dtype=float), np.asarray(weights, dtype=float)


def _small_feature_value(
    composition: Composition,
    property_name: str | None,
    statistic: str,
    pattern: str,
) -> float:
    if statistic == "entropy":
        allowed = _small_composition_groups(composition)[pattern]
        amounts = [float(amount) for elem, amount in composition.items() if elem.symbol in allowed]
        if len(amounts) <= 1:
            return 0.0
        fractions = np.asarray(amounts, dtype=float) / sum(amounts)
        return float(-np.sum(fractions * np.log(fractions)))

    values, weights = _small_weighted_values(composition, str(property_name), pattern)
    if values.size == 0:
        return 0.0
    weights /= weights.sum()
    if statistic == "mean":
        return float(np.average(values, weights=weights))
    if statistic == "mean_square":
        return float(np.average(values**2, weights=weights))
    if statistic == "variance":
        mean = np.average(values, weights=weights)
        return float(np.average((values - mean) ** 2, weights=weights))
    if statistic == "geometric_mean":
        if np.any(values <= 0):
            return 0.0
        return float(np.exp(np.average(np.log(values), weights=weights)))
    raise ValueError(f"Unsupported Small feature statistic: {statistic}")


def small_composition_features(formula: str) -> pd.Series:
    """Compute the selected non-redundant Small/Kong composition features."""
    composition = Composition(formula)
    return pd.Series({
        column: _small_feature_value(composition, property_name, statistic, pattern)
        for column, property_name, statistic, pattern in SMALL_FEATURE_SPECS
    })


def normalize_family(value) -> str:
    """Normalize family labels into stable category names."""
    if pd.isna(value):
        return "unknown"
    text = str(value).strip().lower()
    if not text or text in {"nan", "none", "unknown", "null"}:
        return "unknown"
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    text = text or "unknown"
    aliases = {
        "argyrodite": "argyrodites",
        "argyrodite_like": "argyrodites",
        "lgps_like": "lgps",
        "halide": "halides",
        "halide_like": "halides",
        "oxyhalide": "halides",
        "halide_oxyhalide": "halides",
    }
    return aliases.get(text, text)


def parse_conductivity(value):
    """Parse conductivity value and qualifier."""
    if pd.isna(value):
        return math.nan, "missing"
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value), "exact"

    text = str(value).strip().replace("−", "-")
    qualifier = "exact"
    if text.startswith(("<", "≤")):
        qualifier = "upper_bound"
    elif text.startswith((">", "≥")):
        qualifier = "lower_bound"

    pattern = re.compile(r"^[<>=~≤≥\s]*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)")
    match = pattern.match(text.replace(",", ""))
    if not match:
        return math.nan, "unparsed"
    return float(match.group(1)), qualifier


REDUNDANT_FEATURES = {
    "ρ⁺(incl Li⁺) - ρ⁻",
    "r⁺(excl Li⁺) / r⁻",
    "ρₐₗₗ (C m⁻³)",
    "χ⁺(incl Li⁺)",
    "nₐₙᵢₒₙ",
    "nₕₐₗᵢdₑ",
}


@dataclass
class FeatureConfig:
    """Options for conductivity filtering and feature construction.

    min_conductivity:
        Minimum conductivity kept in the training feature table. The default
        1e-6 reproduces the current main training policy. Use None to keep all
        rows, especially for prediction inputs with dummy conductivity values.
    include_family:
        If True, add one numeric "family" feature encoded from the text Family
        column. Family aliases such as argyrodite_like are folded into the core
        family name before encoding.
    include_interactions:
        If True, add manually defined interaction descriptors derived from the
        base composition descriptors.
    include_small_features:
        If True, add the selected Small/Kong composition descriptors.
    drop_redundant:
        If True, exclude REDUNDANT_FEATURES from the model feature list. The
        columns remain in the output table but are not used for training.
    family_mapping:
        Optional precomputed text-to-code mapping for family encoding. Pass the
        training mapping during prediction to keep numeric codes consistent.
    output_path:
        Feature CSV path. Use None to skip automatic file writing.
    """

    min_conductivity: float | None = 1e-6
    include_family: bool = True
    include_interactions: bool = True
    include_small_features: bool = True
    drop_redundant: bool = True
    family_mapping: dict[str, int] | None = None
    output_path: Path | None = None


@dataclass
class FeatureResult:
    table: pd.DataFrame
    removed: pd.DataFrame
    feature_columns: list[str]
    family_mapping: dict[str, int]
    summary: dict

    def to_file(self, path: str | Path | None = None) -> None:
        output_path = Path(path or DEFAULT_FEATURE_PATH)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.table.to_csv(output_path, index=False)


def family_code_mapping(labels: pd.Series, existing: dict[str, int] | None = None) -> dict[str, int]:
    if existing is not None:
        return dict(existing)
    normalized = labels.apply(normalize_family)
    families = sorted(label for label in normalized.dropna().unique() if label != "unknown")
    return {"unknown": 0, **{label: index for index, label in enumerate(families, start=1)}}


def encode_family(labels: pd.Series, mapping: dict[str, int]) -> pd.Series:
    normalized = labels.apply(normalize_family)
    return normalized.map(mapping).fillna(0).astype(float)


def _apply_conductivity_policy(frame: pd.DataFrame, config: FeatureConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    values = frame[TARGET_COLUMN].apply(parse_conductivity)
    frame = frame.copy()
    frame["conductivity_value"] = [item[0] for item in values]
    frame["conductivity_qualifier"] = [item[1] for item in values]
    removed = pd.DataFrame(columns=[*frame.columns, "removed_reason"])

    is_upper = frame["conductivity_qualifier"].eq("upper_bound")
    conductivity = pd.to_numeric(frame["conductivity_value"], errors="coerce")

    if config.min_conductivity is not None:
        remove_mask = (conductivity < config.min_conductivity) | (
            is_upper & (conductivity <= config.min_conductivity)
        )
        removed = frame[remove_mask].copy()
        if not removed.empty:
            removed["removed_reason"] = f"filtered_out: conductivity < {config.min_conductivity:g}"
        frame = frame[~remove_mask].copy()

    frame["conductivity_used"] = frame["conductivity_value"]

    frame["sample_weight"] = 1.0
    frame["log10_conductivity"] = np.log10(
        pd.to_numeric(frame["conductivity_used"], errors="coerce").where(
            pd.to_numeric(frame["conductivity_used"], errors="coerce") > 0
        )
    )
    return frame.reset_index(drop=True), removed.reset_index(drop=True)


def _add_composition_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    removed = []
    total = len(frame)
    for index, (_, row) in enumerate(frame.iterrows(), start=1):
        try:
            rows.append(composition_features(row["True Composition"]))
        except Exception as exc:
            removed_row = row.to_dict()
            removed_row["removed_reason"] = f"feature_error: {exc}"
            removed.append(removed_row)
            rows.append(pd.Series(dtype=object))
        if index == 1 or index % 50 == 0 or index == total:
            print(f"Computed composition features {index}/{total}", flush=True)

    feature_frame = pd.DataFrame(rows)
    output = pd.concat([frame.reset_index(drop=True), feature_frame], axis=1)
    error_removed = pd.DataFrame(removed)
    if not error_removed.empty:
        error_ids = set(error_removed["ID"].astype(str))
        output = output[~output["ID"].astype(str).isin(error_ids)].copy()
    return output.reset_index(drop=True), error_removed.reset_index(drop=True)


def _add_family_feature(frame: pd.DataFrame, config: FeatureConfig) -> tuple[pd.DataFrame, dict[str, int]]:
    frame = frame.copy()
    labels = (
        frame[FAMILY_COLUMN]
        if FAMILY_COLUMN in frame.columns
        else pd.Series("unknown", index=frame.index)
    )
    normalized = labels.apply(normalize_family)
    mapping = family_code_mapping(normalized, config.family_mapping)
    frame[FAMILY_COLUMN] = normalized
    if config.include_family:
        frame[FAMILY_FEATURE_COLUMN] = encode_family(normalized, mapping)
    return frame, mapping


def _add_interactions(frame: pd.DataFrame, include: bool) -> pd.DataFrame:
    if not include:
        return frame
    frame = frame.copy()
    if "r⁺(incl Li⁺) / r⁻" in frame.columns and "χ⁺(incl Li⁺) - χ⁻" in frame.columns:
        frame["r_ratio_x_chi_diff"] = (
            pd.to_numeric(frame["r⁺(incl Li⁺) / r⁻"], errors="coerce")
            * pd.to_numeric(frame["χ⁺(incl Li⁺) - χ⁻"], errors="coerce")
        )
    if "ρ⁺(incl Li⁺) / ρ⁻" in frame.columns:
        rho_ratio = pd.to_numeric(frame["ρ⁺(incl Li⁺) / ρ⁻"], errors="coerce")
        frame["log_rho_ratio"] = np.log10(rho_ratio.clip(lower=1e-3))
    if "n_Li" in frame.columns and "r⁻ (pm)" in frame.columns:
        frame["n_Li × r⁻ (pm)"] = (
            pd.to_numeric(frame["n_Li"], errors="coerce")
            * pd.to_numeric(frame["r⁻ (pm)"], errors="coerce")
        )
    if "χₘₐₓ - χₘᵢₙ" in frame.columns and "rₐₗₗ (pm)" in frame.columns:
        frame["chi_range_x_r_avg"] = (
            pd.to_numeric(frame["χₘₐₓ - χₘᵢₙ"], errors="coerce")
            * pd.to_numeric(frame["rₐₗₗ (pm)"], errors="coerce")
        )
    if "Φ⁺(incl Li⁺) (|Z| pm⁻¹)" in frame.columns and "r⁺(incl Li⁺) - r⁻" in frame.columns:
        frame["field_x_r_diff"] = (
            pd.to_numeric(frame["Φ⁺(incl Li⁺) (|Z| pm⁻¹)"], errors="coerce")
            * pd.to_numeric(frame["r⁺(incl Li⁺) - r⁻"], errors="coerce")
        )
    return frame


def _add_small_features(frame: pd.DataFrame, include: bool) -> pd.DataFrame:
    if not include:
        return frame
    small_rows = frame["True Composition"].apply(small_composition_features)
    return pd.concat([frame.reset_index(drop=True), small_rows.reset_index(drop=True)], axis=1)


def infer_feature_columns(frame: pd.DataFrame, *, drop_redundant: bool = True) -> list[str]:
    excluded = set(METADATA_COLUMNS)
    excluded.add("Z_by_element")
    if drop_redundant:
        excluded.update(REDUNDANT_FEATURES)
    columns = []
    for column in frame.columns:
        if column in excluded:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        if pd.api.types.is_numeric_dtype(values) and values.notna().any():
            columns.append(column)
    return columns


def make_feature_table(
    df: pd.DataFrame,
    config: FeatureConfig | None = None,
) -> FeatureResult:
    """Build descriptors, log10 target, family encoding, and feature columns."""

    config = config or FeatureConfig()
    filtered, low_removed = _apply_conductivity_policy(df, config)
    featured, feature_removed = _add_composition_features(filtered)
    featured, family_mapping = _add_family_feature(featured, config)
    featured = _add_interactions(featured, config.include_interactions)
    featured = _add_small_features(featured, config.include_small_features)
    feature_columns = infer_feature_columns(featured, drop_redundant=config.drop_redundant)

    removed = pd.concat([low_removed, feature_removed], ignore_index=True)
    summary = {
        "config": asdict(config) | {"family_mapping": family_mapping},
        "input_rows": int(len(df)),
        "output_rows": int(len(featured)),
        "removed_rows": int(len(removed)),
        "feature_count": int(len(feature_columns)),
        "feature_columns": feature_columns,
        "family_mapping": family_mapping,
        "target_log10": {
            "min": float(featured["log10_conductivity"].min()),
            "median": float(featured["log10_conductivity"].median()),
            "max": float(featured["log10_conductivity"].max()),
        },
    }
    result = FeatureResult(
        table=featured,
        removed=removed,
        feature_columns=feature_columns,
        family_mapping=family_mapping,
        summary=summary,
    )
    if config.output_path is not None:
        result.to_file(config.output_path)
    return result
