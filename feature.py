from __future__ import annotations
import json
import math
from functools import lru_cache
from itertools import product
from pathlib import Path
import pandas as pd
from mendeleev import element
from pymatgen.core import Composition
from pymatgen.core.periodic_table import Element

INPUT_CSV = Path("rawdata/all.csv")
OUTPUT_CSV = Path("rawdata/all_selected_features.csv")
OXIDATION_CONFIG = Path("config/oxidation_states.json")
ORGANIC_MARKER_ELEMENTS = frozenset({"C", "H"})
ORGANIC_NEUTRAL_ELEMENTS = frozenset({"C", "H", "N", "O"})
ELEMENTARY_CHARGE_C = 1.602e-19
PM_TO_M = 1e-12
CHARGE_RESIDUAL_LIMIT = 1.0
HALIDE_ELEMENTS = frozenset({"F", "Cl", "Br", "I"})
ANION_ELEMENTS = frozenset({"O", "S", "Se", "Te", "N", "P", "F", "Cl", "Br", "I"})
BASE_COLUMNS = [
    "ID",
    "True Composition",
    "Ionic conductivity (S cm-1)",
]


# ---------------------------------------------------------------------------
# 电负性
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def pauling_electronegativity(symbol: str) -> float | None:
    value = element(symbol).electronegativity("pauling")
    return None if value is None else float(value)


# ---------------------------------------------------------------------------
# 氧化态配置 & 猜测
# ---------------------------------------------------------------------------

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
        state_penalty = 0.001 * sum(
            abs(float(state))
            for state in states
        )
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


def valence_residual(
    formula: str,
    amounts: dict[str, float],
    oxidation_guess: dict[str, float],
) -> float:
    return charge_residual(amounts, oxidation_guess)


def charge_balance_failures(
    formulas: pd.Series,
    ids: pd.Series | None = None,
) -> list[dict[str, object]]:
    failures: list[dict[str, object]] = []
    for index, formula in formulas.items():
        composition = Composition(formula)
        amounts = {
            symbol: float(composition.get_el_amt_dict()[symbol])
            for symbol in (elem.symbol for elem in composition.elements)
        }
        oxidation_guesses, _, _, _ = oxidation_state_guesses(composition)
        oxidation_guess = oxidation_guesses[0] if oxidation_guesses else {}
        residual = charge_residual(amounts, oxidation_guess)
        if abs(residual) >= CHARGE_RESIDUAL_LIMIT:
            failure: dict[str, object] = {
                "formula": formula,
                "residual_charge": residual,
                "oxidation_guess": element_value_list(oxidation_guess),
            }
            if ids is not None:
                failure["ID"] = ids.loc[index]
            failures.append(failure)
    return failures


# ---------------------------------------------------------------------------
# 离子半径
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def ionic_radius_pm(symbol: str, charge: float) -> float | None:
    if charge == 0 or not float(charge).is_integer():
        return None
    charge_int = int(charge)
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


# ---------------------------------------------------------------------------
# 辅助计算
# ---------------------------------------------------------------------------

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
    radius_m = radius_pm * PM_TO_M
    return 3 * abs(charge) * ELEMENTARY_CHARGE_C / (4 * math.pi * radius_m**3)


def ionic_potential(charge: float, radius_pm: float | None) -> float | None:
    if radius_pm is None:
        return None
    return abs(charge) / radius_pm


def element_value_list(values: dict[str, object]) -> str:
    items = [
        {"element": symbol, "value": value}
        for symbol, value in values.items()
    ]
    return json.dumps(items, ensure_ascii=False)


# ---------------------------------------------------------------------------
# 分类
# ---------------------------------------------------------------------------

def classify_elements(
    elements: list[str],
    oxidation_guess: dict[str, float],
) -> dict[str, str]:
    classification: dict[str, str] = {}
    for symbol in elements:
        if symbol == "Li":
            classification[symbol] = "Li"
        elif symbol == "H" and oxidation_guess.get(symbol, 0) > 0:
            classification[symbol] = "proton"
        elif symbol in HALIDE_ELEMENTS:
            classification[symbol] = "halide"
        elif oxidation_guess.get(symbol, 0) < 0 or symbol in ANION_ELEMENTS:
            classification[symbol] = "anion"
        else:
            classification[symbol] = "host_cation"
    return classification


# ---------------------------------------------------------------------------
# 每行特征提取
# ---------------------------------------------------------------------------

def composition_features(formula: str) -> pd.Series:
    composition = Composition(formula)
    elements = [elem.symbol for elem in composition.elements]
    amounts = {
        symbol: float(composition.get_el_amt_dict()[symbol])
        for symbol in elements
    }
    total_atoms = sum(amounts.values())

    # 氧化态猜测
    oxidation_guesses, _, _, _ = oxidation_state_guesses(composition)
    ox = oxidation_guesses[0] if oxidation_guesses else {}
    residual = valence_residual(formula, amounts, ox)

    # 每元素基础属性
    en_map: dict[str, float | None] = {}
    r_map: dict[str, float | None] = {}
    rho_map: dict[str, float | None] = {}
    phi_map: dict[str, float | None] = {}
    for s in elements:
        en_map[s] = pauling_electronegativity(s)
        charge = ox.get(s)
        if charge is not None:
            r_map[s] = ionic_radius_pm(s, charge)
            rho_map[s] = charge_density_c_m3(charge, r_map[s])
            phi_map[s] = ionic_potential(charge, r_map[s])
        else:
            r_map[s] = None
            rho_map[s] = None
            phi_map[s] = None

    # 分类
    cls = classify_elements(elements, ox)
    anion_sym = [s for s in elements if cls.get(s) in ("anion", "halide")]
    cat_incl_sym = [s for s in elements if cls.get(s) in ("Li", "host_cation")]
    cat_excl_sym = [s for s in elements if cls.get(s) == "host_cation"]
    halide_sym = [s for s in elements if cls.get(s) == "halide"]
    non_halide_anion_sym = [s for s in elements if cls.get(s) == "anion"]

    # 基础平均值
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

    # χ max - χ min
    en_values = [v for v in en_map.values() if v is not None]
    chi_max_min = (max(en_values) - min(en_values)) if len(en_values) >= 2 else None

    return pd.Series(
        {
            # --- 氧化态 Z ---
            "Z_by_element": element_value_list(ox),
            "valence_residual": residual,
            # --- 电负性 χ ---
            "χ_all": chi_all,
            "χ-": chi_minus,
            "χ+(incl Li+)": chi_plus_incl,
            "χ+(excl Li+)": chi_plus_excl,
            "χ+(incl Li+) - χ-": safe_diff(chi_plus_incl, chi_minus),
            "χ+(excl Li+) - χ-": safe_diff(chi_plus_excl, chi_minus),
            "χ_max - χ_min": chi_max_min,
            # --- 半径 r ---
            "r_all (pm)": r_all,
            "r- (pm)": r_minus,
            "r+(incl Li+) (pm)": r_plus_incl,
            "r+(excl Li+) (pm)": r_plus_excl,
            "r+(excl Li+) - r-": safe_diff(r_plus_excl, r_minus),
            "r+(incl Li+) - r-": safe_diff(r_plus_incl, r_minus),
            "r+(excl Li+) / r-": safe_ratio(r_plus_excl, r_minus),
            "r+(incl Li+) / r-": safe_ratio(r_plus_incl, r_minus),
            # --- 电荷密度 ρ ---
            "ρ_all (C m⁻³)": rho_all,
            "ρ- (C m⁻³)": rho_minus,
            "ρ+(incl Li+) (C m⁻³)": rho_plus_incl,
            "ρ+(excl Li+) (C m⁻³)": rho_plus_excl,
            "ρ+(incl Li+) / ρ-": safe_ratio(rho_plus_incl, rho_minus),
            "ρ+(incl Li+) - ρ-": safe_diff(rho_plus_incl, rho_minus),
            # --- 离子势 Φ（仅阳离子）---
            "Φ+(incl Li+) (|Z| pm⁻¹)": phi_plus_incl,
            "Φ+(excl Li+) (|Z| pm⁻¹)": phi_plus_excl,
            # --- 组分占比 n ---
            "n_halide": (
                sum(amounts.get(s, 0) for s in halide_sym) / total_atoms
                if total_atoms > 0 else None
            ),
            "n_host_cation": (
                sum(amounts.get(s, 0) for s in cat_excl_sym) / total_atoms
                if total_atoms > 0 else None
            ),
            "n_anion": (
                sum(amounts.get(s, 0) for s in non_halide_anion_sym) / total_atoms
                if total_atoms > 0 else None
            ),
            "n_Li": (
                amounts.get("Li", 0) / total_atoms
                if total_atoms > 0 else None
            ),
        }
    )


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main() -> None:
    df = pd.read_csv(INPUT_CSV)
    selected = df.loc[:, BASE_COLUMNS].copy()

    # 删除含有机物的行
    organic_mask = selected["True Composition"].apply(
        lambda f: contains_organic_molecule(Composition(f))
    )
    removed_count = organic_mask.sum()
    selected = selected[~organic_mask].reset_index(drop=True)
    print(f"Removed {removed_count} organic-containing rows")

    failures = charge_balance_failures(
        selected["True Composition"],
        selected["ID"],
    )
    if failures:
        print(
            "Warning: charge balance residual is high for "
            f"{len(failures)} rows with abs(residual_charge) >= "
            f"{CHARGE_RESIDUAL_LIMIT:g}:"
        )
        for failure in failures:
            print(
                f"ID={failure.get('ID', '')}; "
                f"formula={failure['formula']}; "
                f"residual_charge={failure['residual_charge']:.6g}; "
                f"oxidation_guess={failure['oxidation_guess']}"
            )

    total = len(selected)
    start = pd.Timestamp.now()
    features = []
    for index, (_, row) in enumerate(selected.iterrows(), start=1):
        features.append(
            composition_features(row["True Composition"])
        )
        if index == 1 or index % 50 == 0 or index == total:
            elapsed = (pd.Timestamp.now() - start).total_seconds()
            seconds_per_row = elapsed / index
            remaining = max(total - index, 0) * seconds_per_row
            print(
                f"Processed {index}/{total} rows; "
                f"elapsed {elapsed:.1f}s; ETA {remaining:.1f}s",
                flush=True,
            )
    feature_df = pd.DataFrame(features)
    output = pd.concat(
        [
            selected[["ID", "True Composition"]],
            feature_df,
            selected[["Ionic conductivity (S cm-1)"]],
        ],
        axis=1,
    )
    output.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {len(output)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
