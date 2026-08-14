"""Build the fixed composition-difference feature set for trend modeling.

The training input is the emitted pair table.  Prediction input only needs an
ordered formula pair (A -> B) and an optional family label.  Pair metadata and
conductivity columns are retained for audit and grouping, but are not model
features.
"""

from __future__ import annotations

# Allow running this module directly as a script (VS Code "Run" button): set
# __package__ so the relative imports below resolve, and expose project root.
if __package__ is None:
    import sys
    from pathlib import Path as _Path
    _FILE = _Path(__file__).resolve()
    if str(_FILE.parents[2]) not in sys.path:
        sys.path.insert(0, str(_FILE.parents[2]))
    __package__ = f"{_FILE.parents[1].name}.{_FILE.parents[0].name}"

from collections.abc import Iterable
import json
import math
from pathlib import Path
import time

import numpy as np
import pandas as pd
from pymatgen.core import Composition

# HOOK for reusing additional absolute-pipeline features in trend: to add more
# absolute descriptors later, extend this import and MODEL_FEATURE_COLUMNS, then
# retrain. No dedicated placeholder interface is needed.
from ..features import composition_features, normalize_family, small_composition_features
from ..paths import TREND_DIR


# Point-run defaults (compact pairs chain).
PAIRS_INPUT = TREND_DIR / "data-trend-v1-pairs.csv"
DEFAULT_PAIRS_FEATURE = TREND_DIR / "data-trend-v2-pairs-feature.csv"

# Legacy full-descriptor feature file paths (kept for library API stability).
DEFAULT_PAIR_INPUT = TREND_DIR / "data-labed-v1.csv"
DEFAULT_TRAIN_OUTPUT = TREND_DIR / "data-trend-v1-features.csv"
DEFAULT_PREDICTION_OUTPUT = TREND_DIR / "data-trend-v1-prediction-features.csv"
DEFAULT_FEATURE_LIST_OUTPUT = TREND_DIR / "data-trend-v2-feature-list.txt"
DEFAULT_SCHEMA_OUTPUT = TREND_DIR / "data-trend-v2-feature-schema.json"


# Retained from the previous CatBoost run at >=2% built-in importance.
SIGNED_DELTA_FEATURES = [
    "delta_entropy_c_l",
    "delta_log10_rho_plus_incl_li",
    "delta_field_x_r_diff",
    "delta_n_Li",
    "delta_log_rho_ratio",
    "delta_r_all_pm",
    "delta_ir_mean_square_all",
    "delta_atwt_geometric_mean_all",
    "delta_chi_all",
]

MEAN_FEATURES = [
    "mean_entropy_c_l",
    "mean_log10_rho_plus_incl_li",
    "mean_log10_phi_plus_incl_li",
    "mean_field_x_r_diff",
    "mean_n_Li",
    "mean_r_all_pm",
    "mean_atwt_geometric_mean_all",
]

MAGNITUDE_FEATURES = [
    "abs_delta_entropy_c_l",
    "abs_delta_log10_rho_plus_incl_li",
    "abs_delta_log10_phi_plus_incl_li",
    "abs_delta_field_x_r_diff",
    "composition_L1_distance",
]

# Consensus top descriptors from the historical absolute 26-feature baseline.
# Each contributes the directed A baseline and B-A change; together they retain
# both intrinsic material context and modification information.
SELECTED_ABSOLUTE_DESCRIPTORS = [
    "r_all_pm",
    "chi_plus_excl_minus_chi_minus",
    "chi_all",
    "phi_plus_excl_li",
    "field_x_r_diff",
    "r_plus_incl_minus_r_minus",
    "chi_plus_incl_minus_chi_minus",
    "chi_range_x_r_avg",
    "r_plus_incl_li_pm",
    "rho_plus_excl_li_c_m3",
    "r_plus_excl_minus_r_minus",
    "n_li_x_r_minus_pm",
]
ABSOLUTE_DELTA_BY_DESCRIPTOR = {
    "r_all_pm": "delta_r_all_pm",
    "chi_all": "delta_chi_all",
    "field_x_r_diff": "delta_field_x_r_diff",
    **{
        descriptor: f"delta_{descriptor}"
        for descriptor in SELECTED_ABSOLUTE_DESCRIPTORS
        if descriptor not in {"r_all_pm", "chi_all", "field_x_r_diff"}
    },
}
ABSOLUTE_DELTA_FEATURES = list(ABSOLUTE_DELTA_BY_DESCRIPTOR.values())
A_BASELINE_FEATURES = [
    f"a_{descriptor}" for descriptor in SELECTED_ABSOLUTE_DESCRIPTORS
]
B_BASELINE_FEATURES = [
    f"b_{descriptor}" for descriptor in SELECTED_ABSOLUTE_DESCRIPTORS
]
SIGNED_DELTA_FEATURES.extend(
    feature
    for feature in ABSOLUTE_DELTA_FEATURES
    if feature not in SIGNED_DELTA_FEATURES
)

MODEL_FEATURE_COLUMNS = [
    *SIGNED_DELTA_FEATURES,
    *MEAN_FEATURES,
    *MAGNITUDE_FEATURES,
    *A_BASELINE_FEATURES,
    *B_BASELINE_FEATURES,
]

# Kept computable so previously trained artifacts remain loadable for
# comparison, although new feature tables only emit MODEL_FEATURE_COLUMNS.
LEGACY_MODEL_FEATURE_COLUMNS = [
    "delta_entropy_c_l",
    "delta_log10_rho_plus_incl_li",
    "delta_log10_phi_plus_incl_li",
    "delta_field_x_r_diff",
    "delta_n_Li",
    "delta_log_rho_ratio",
    "delta_r_all_pm",
    "delta_n_host_cation",
    "delta_ir_mean_square_all",
    "delta_atwt_geometric_mean_all",
    "delta_chi_all",
    "delta_chi_minus",
    "delta_r_minus_pm",
    "mean_entropy_c_l",
    "mean_log10_rho_plus_incl_li",
    "mean_log10_phi_plus_incl_li",
    "mean_field_x_r_diff",
    "mean_n_Li",
    "mean_r_all_pm",
    "mean_atwt_geometric_mean_all",
    "mean_chi_all",
    "abs_delta_entropy_c_l",
    "abs_delta_log10_rho_plus_incl_li",
    "abs_delta_log10_phi_plus_incl_li",
    "abs_delta_field_x_r_diff",
    "abs_delta_n_Li",
    "composition_L1_distance",
]
ALL_COMPUTED_FEATURE_COLUMNS = list(dict.fromkeys([
    *MODEL_FEATURE_COLUMNS,
    *LEGACY_MODEL_FEATURE_COLUMNS,
]))
OPTIONAL_MODEL_FEATURE_COLUMNS = set([
    *ABSOLUTE_DELTA_FEATURES,
    *A_BASELINE_FEATURES,
    *B_BASELINE_FEATURES,
])

TREND_ABSOLUTE_THRESHOLD_S_CM = 1e-4
# Backward-compatible alias for callers of the compact-pair API.
SIMPLE_ABSOLUTE_THRESHOLD_S_CM = TREND_ABSOLUTE_THRESHOLD_S_CM


# Internal descriptor names are deliberately separate from emitted model names.
_DESCRIPTOR_COLUMNS = {
    "entropy_c_l",
    "log10_rho_plus_incl_li",
    "log10_phi_plus_incl_li",
    "field_x_r_diff",
    "n_Li",
    "log_rho_ratio",
    "r_all_pm",
    "n_host_cation",
    "ir_mean_square_all",
    "atwt_geometric_mean_all",
    "chi_all",
    "chi_minus",
    "r_minus_pm",
    *SELECTED_ABSOLUTE_DESCRIPTORS,
}


def classify_trend_delta(
    delta_s_cm: object,
    *,
    threshold_s_cm: float = TREND_ABSOLUTE_THRESHOLD_S_CM,
) -> np.ndarray:
    """Classify conductivity deltas with the shared absolute-change policy."""
    if threshold_s_cm != TREND_ABSOLUTE_THRESHOLD_S_CM:
        raise ValueError(
            "Trend labels require the fixed absolute threshold "
            f"{TREND_ABSOLUTE_THRESHOLD_S_CM:g} S/cm."
        )
    delta = np.asarray(delta_s_cm, dtype=float)
    if not np.isfinite(delta).all():
        raise ValueError("Conductivity deltas must be finite.")
    return np.where(
        delta > threshold_s_cm,
        "increase",
        np.where(delta < -threshold_s_cm, "decrease", "unchanged"),
    )


# These columns are retained as training metadata/audit, never inferred by
# selecting all numeric columns.
TRACE_COLUMNS = [
    "pair_id",
    "group_id",
    "group_label",
    "group_size",
    "id_a",
    "id_b",
    "source_row_a",
    "source_row_b",
    "formula_a",
    "formula_b",
    "doi",
    "temperature",
    "temperature_C",
    "temperature_basis",
    "temperature_raw",
    "preparation_method",
    "family",
    "family_normalized",
    "amorphous_status_a",
    "amorphous_status_b",
    "crystal_phase_a",
    "crystal_phase_b",
    "phase_quality_status_a",
    "phase_quality_status_b",
    "conductivity_measurement_type",
    "conductivity_measurement_type_a",
    "conductivity_measurement_type_b",
    "measurement_quality_a",
    "measurement_quality_b",
    "measurement_quality_flags_a",
    "measurement_quality_flags_b",
    "source_curation_status_a",
    "source_curation_status_b",
    "source_curation_flags_a",
    "source_curation_flags_b",
    "year_a",
    "year_b",
    "label_quality",
    "pair_weight_group_equal",
]

LABEL_COLUMNS = [
    "trend_label",
    "trend_class",
    "电导率变化趋势",
    "direction_label",
    "direction_class",
]

AUDIT_COLUMNS = [
    "conductivity_a_S_cm-1",
    "conductivity_b_S_cm-1",
    "conductivity_qualifier_a",
    "conductivity_qualifier_b",
    "conductivity_source_a",
    "conductivity_source_b",
    "conductivity_parse_status_a",
    "conductivity_parse_status_b",
    "conductivity_unit_a",
    "conductivity_unit_b",
    "conductivity_value_text_a",
    "conductivity_value_text_b",
    "delta_conductivity_S_cm-1",
    "absolute_change_conductivity_S_cm-1",
    "conductivity_ratio_b_over_a",
    "conductivity_fold_change",
    "absolute_change_met",
    "fold_change_met",
    "change_trigger",
    "absolute_change_threshold_S_cm-1",
    "fold_change_threshold",
    "temperature_consistency_status_a",
    "temperature_consistency_status_b",
]


class FeatureComputationError(ValueError):
    """Raised when a formula cannot produce a valid selected descriptor."""


def _as_float(value: object, *, formula: str, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise FeatureComputationError(
            f"Descriptor {name!r} is not numeric for formula {formula!r}: {value!r}"
        ) from exc
    if not math.isfinite(number):
        raise FeatureComputationError(
            f"Descriptor {name!r} is not finite for formula {formula!r}: {number!r}"
        )
    return number


def _as_optional_float(value: object, *, formula: str, name: str) -> float:
    """Return NaN when a descriptor is chemically undefined."""
    if value is None or pd.isna(value):
        return math.nan
    return _as_float(value, formula=formula, name=name)


def _positive_log10(value: object, *, formula: str, name: str) -> float:
    number = _as_float(value, formula=formula, name=name)
    if number <= 0:
        raise FeatureComputationError(
            f"Descriptor {name!r} must be positive before log10 for formula {formula!r}: {number}"
        )
    return float(math.log10(number))


def _composition_l1_distance(formula_a: str, formula_b: str) -> float:
    try:
        composition_a = Composition(formula_a).fractional_composition
        composition_b = Composition(formula_b).fractional_composition
    except Exception as exc:
        raise FeatureComputationError(
            f"Cannot parse formula pair for composition distance: {formula_a!r}, {formula_b!r}"
        ) from exc
    amounts_a = composition_a.get_el_amt_dict()
    amounts_b = composition_b.get_el_amt_dict()
    symbols = set(amounts_a) | set(amounts_b)
    distance = sum(
        abs(float(amounts_b.get(symbol, 0.0)) - float(amounts_a.get(symbol, 0.0)))
        for symbol in symbols
    )
    if not math.isfinite(distance) or distance < 0 or distance > 2.0 + 1e-9:
        raise FeatureComputationError(
            f"Invalid composition L1 distance for {formula_a!r}, {formula_b!r}: {distance}"
        )
    return float(distance)


def compute_formula_descriptor(formula: str) -> dict[str, float]:
    """Compute the scalar descriptors needed for one formula."""

    formula = str(formula).strip()
    if not formula:
        raise FeatureComputationError("Formula is empty.")
    try:
        base = composition_features(formula)
        small = small_composition_features(formula)
    except Exception as exc:
        raise FeatureComputationError(
            f"Cannot compute composition descriptors for formula {formula!r}: {exc}"
        ) from exc

    rho_plus = _positive_log10(
        base["ρ⁺(incl Li⁺) (C m⁻³)"],
        formula=formula,
        name="ρ⁺(incl Li⁺)",
    )
    phi_plus = _positive_log10(
        base["Φ⁺(incl Li⁺) (|Z| pm⁻¹)"],
        formula=formula,
        name="Φ⁺(incl Li⁺)",
    )
    rho_ratio = _as_float(
        base["ρ⁺(incl Li⁺) / ρ⁻"],
        formula=formula,
        name="ρ⁺(incl Li⁺) / ρ⁻",
    )
    # Match main.features._add_interactions exactly: this source descriptor is
    # already log10-transformed and must not be logged a second time later.
    log_rho_ratio = float(math.log10(max(rho_ratio, 1e-3)))
    phi_incl = _as_float(
        base["Φ⁺(incl Li⁺) (|Z| pm⁻¹)"],
        formula=formula,
        name="Φ⁺(incl Li⁺)",
    )
    radius_difference = (
        _as_float(
            base["r⁺(incl Li⁺) (pm)"],
            formula=formula,
            name="r⁺(incl Li⁺)",
        )
        - _as_float(
            base["r⁻ (pm)"],
            formula=formula,
            name="r⁻",
        )
    )
    r_all = _as_float(base["rₐₗₗ (pm)"], formula=formula, name="rₐₗₗ")
    r_minus = _as_float(base["r⁻ (pm)"], formula=formula, name="r⁻")
    n_li = _as_float(base["n_Li"], formula=formula, name="n_Li")
    chi_range = _as_float(
        base["χₘₐₓ - χₘᵢₙ"], formula=formula, name="χₘₐₓ - χₘᵢₙ"
    )

    descriptor = {
        "entropy_c_l": _as_float(
            small["entropy_c_l"], formula=formula, name="entropy_c_l"
        ),
        "log10_rho_plus_incl_li": rho_plus,
        "log10_phi_plus_incl_li": phi_plus,
        "field_x_r_diff": phi_incl * radius_difference,
        "n_Li": n_li,
        "log_rho_ratio": log_rho_ratio,
        "r_all_pm": r_all,
        "n_host_cation": _as_float(
            base["nₕₒₛₜ cₐₜᵢₒₙ"], formula=formula, name="nₕₒₛₜ cₐₜᵢₒₙ"
        ),
        "ir_mean_square_all": _as_float(
            small["ir_mean_square_all"],
            formula=formula,
            name="ir_mean_square_all",
        ),
        "atwt_geometric_mean_all": _as_float(
            small["atwt_geometric_mean_all"],
            formula=formula,
            name="atwt_geometric_mean_all",
        ),
        "chi_all": _as_float(
            base["χₐₗₗ"], formula=formula, name="χₐₗₗ"
        ),
        "chi_minus": _as_float(
            base["χ⁻"], formula=formula, name="χ⁻"
        ),
        "r_minus_pm": r_minus,
        "chi_plus_excl_minus_chi_minus": _as_optional_float(
            base["χ⁺(excl Li⁺) - χ⁻"], formula=formula,
            name="χ⁺(excl Li⁺) - χ⁻",
        ),
        "phi_plus_excl_li": _as_optional_float(
            base["Φ⁺(excl Li⁺) (|Z| pm⁻¹)"], formula=formula,
            name="Φ⁺(excl Li⁺)",
        ),
        "r_plus_incl_minus_r_minus": _as_float(
            base["r⁺(incl Li⁺) - r⁻"], formula=formula,
            name="r⁺(incl Li⁺) - r⁻",
        ),
        "chi_plus_incl_minus_chi_minus": _as_float(
            base["χ⁺(incl Li⁺) - χ⁻"], formula=formula,
            name="χ⁺(incl Li⁺) - χ⁻",
        ),
        "chi_range_x_r_avg": chi_range * r_all,
        "r_plus_incl_li_pm": _as_float(
            base["r⁺(incl Li⁺) (pm)"], formula=formula,
            name="r⁺(incl Li⁺)",
        ),
        "rho_plus_excl_li_c_m3": _as_optional_float(
            base["ρ⁺(excl Li⁺) (C m⁻³)"], formula=formula,
            name="ρ⁺(excl Li⁺)",
        ),
        "r_plus_excl_minus_r_minus": _as_optional_float(
            base["r⁺(excl Li⁺) - r⁻"], formula=formula,
            name="r⁺(excl Li⁺) - r⁻",
        ),
        "n_li_x_r_minus_pm": n_li * r_minus,
    }
    missing = sorted(_DESCRIPTOR_COLUMNS - set(descriptor))
    if missing:
        raise FeatureComputationError(
            f"Internal descriptor mapping is incomplete for {formula!r}: {missing}"
        )
    return descriptor


def _formula_descriptor_cache(
    formulas: Iterable[object],
    *,
    show_progress: bool = False,
) -> dict[str, dict[str, float]]:
    unique_formulas = [
        str(value).strip()
        for value in pd.unique(pd.Series(list(formulas), dtype=object))
    ]
    cache: dict[str, dict[str, float]] = {}
    total = len(unique_formulas)
    start = time.monotonic()
    for index, formula in enumerate(unique_formulas, start=1):
        if formula not in cache:
            cache[formula] = compute_formula_descriptor(formula)
        if show_progress and (index == 1 or index % 50 == 0 or index == total):
            elapsed = time.monotonic() - start
            rate = index / elapsed if elapsed > 0 else 0.0
            eta = (total - index) / rate if rate > 0 else 0.0
            print(
                f"Computed formula descriptors {index}/{total}; "
                f"elapsed={elapsed:.1f}s; eta={eta:.1f}s",
                flush=True,
            )
    return cache


def _pair_numeric_features(
    descriptor_a: dict[str, float],
    descriptor_b: dict[str, float],
    formula_a: str,
    formula_b: str,
) -> dict[str, float]:
    def delta(name: str) -> float:
        return descriptor_b[name] - descriptor_a[name]

    def mean(name: str) -> float:
        return (descriptor_a[name] + descriptor_b[name]) / 2.0

    values = {
        "delta_entropy_c_l": delta("entropy_c_l"),
        "delta_log10_rho_plus_incl_li": delta("log10_rho_plus_incl_li"),
        "delta_log10_phi_plus_incl_li": delta("log10_phi_plus_incl_li"),
        "delta_field_x_r_diff": delta("field_x_r_diff"),
        "delta_n_Li": delta("n_Li"),
        "delta_log_rho_ratio": delta("log_rho_ratio"),
        "delta_r_all_pm": delta("r_all_pm"),
        "delta_n_host_cation": delta("n_host_cation"),
        "delta_ir_mean_square_all": delta("ir_mean_square_all"),
        "delta_atwt_geometric_mean_all": delta("atwt_geometric_mean_all"),
        "delta_chi_all": delta("chi_all"),
        "delta_chi_minus": delta("chi_minus"),
        "delta_r_minus_pm": delta("r_minus_pm"),
        "mean_entropy_c_l": mean("entropy_c_l"),
        "mean_log10_rho_plus_incl_li": mean("log10_rho_plus_incl_li"),
        "mean_log10_phi_plus_incl_li": mean("log10_phi_plus_incl_li"),
        "mean_field_x_r_diff": mean("field_x_r_diff"),
        "mean_n_Li": mean("n_Li"),
        "mean_r_all_pm": mean("r_all_pm"),
        "mean_atwt_geometric_mean_all": mean("atwt_geometric_mean_all"),
        "mean_chi_all": mean("chi_all"),
    }
    values.update(
        {
            "abs_delta_entropy_c_l": abs(values["delta_entropy_c_l"]),
            "abs_delta_log10_rho_plus_incl_li": abs(
                values["delta_log10_rho_plus_incl_li"]
            ),
            "abs_delta_log10_phi_plus_incl_li": abs(
                values["delta_log10_phi_plus_incl_li"]
            ),
            "abs_delta_field_x_r_diff": abs(values["delta_field_x_r_diff"]),
            "abs_delta_n_Li": abs(values["delta_n_Li"]),
            "composition_L1_distance": _composition_l1_distance(
                formula_a, formula_b
            ),
        }
    )
    for descriptor, delta_column in ABSOLUTE_DELTA_BY_DESCRIPTOR.items():
        values[delta_column] = delta(descriptor)
        values[f"a_{descriptor}"] = descriptor_a[descriptor]
        values[f"b_{descriptor}"] = descriptor_b[descriptor]
    for column in ALL_COMPUTED_FEATURE_COLUMNS:
        value = values[column]
        if math.isinf(value) or (
            math.isnan(value) and column not in OPTIONAL_MODEL_FEATURE_COLUMNS
        ):
            raise FeatureComputationError(
                f"Computed feature {column!r} is not finite for pair "
                f"{formula_a!r} -> {formula_b!r}: {value!r}"
            )
    return {
        column: float(values[column])
        for column in ALL_COMPUTED_FEATURE_COLUMNS
    }


def _output_metadata_columns(frame: pd.DataFrame) -> list[str]:
    columns = [column for column in TRACE_COLUMNS if column in frame.columns]
    columns += [column for column in LABEL_COLUMNS if column in frame.columns]
    columns += [column for column in AUDIT_COLUMNS if column in frame.columns]
    return columns


def build_pair_feature_table(pairs: pd.DataFrame) -> pd.DataFrame:
    """Compute model features and retain non-feature training metadata."""

    required = {"pair_id", "group_id", "formula_a", "formula_b"}
    missing = sorted(required - set(pairs.columns))
    if missing:
        raise ValueError(f"Pair table is missing formula columns: {missing}")
    frame = pairs.copy()
    conductivity_columns = {
        "conductivity_a_S_cm-1",
        "conductivity_b_S_cm-1",
    }
    if conductivity_columns <= set(frame.columns):
        conductivity_a = pd.to_numeric(
            frame["conductivity_a_S_cm-1"], errors="raise"
        )
        conductivity_b = pd.to_numeric(
            frame["conductivity_b_S_cm-1"], errors="raise"
        )
        labels = classify_trend_delta(conductivity_b - conductivity_a)
        frame["trend_label"] = labels
        frame["trend_class"] = pd.Series(labels).map(
            {"increase": 1, "decrease": -1, "unchanged": 0}
        ).to_numpy()
    cache = _formula_descriptor_cache(
        pd.concat([frame["formula_a"], frame["formula_b"]], ignore_index=True),
        show_progress=True,
    )
    pair_counts = frame.groupby("group_id", dropna=False)["pair_id"].transform("size")
    feature_records: list[dict[str, float]] = []
    for row in frame.itertuples(index=False):
        formula_a = str(getattr(row, "formula_a")).strip()
        formula_b = str(getattr(row, "formula_b")).strip()
        feature_records.append(
            _pair_numeric_features(
                cache[formula_a],
                cache[formula_b],
                formula_a,
                formula_b,
            )
        )
    feature_frame = pd.DataFrame.from_records(feature_records, columns=MODEL_FEATURE_COLUMNS)

    output = frame.loc[:, _output_metadata_columns(frame)].copy()
    if "family" in output.columns:
        output["family_normalized"] = output["family"].map(normalize_family)
        # Keep the normalized context next to the original family column.
        output = output.loc[
            :, [column for column in TRACE_COLUMNS if column in output.columns]
            + [column for column in LABEL_COLUMNS if column in output.columns]
            + [column for column in AUDIT_COLUMNS if column in output.columns]
        ]
    output = pd.concat(
        [output.reset_index(drop=True), feature_frame.reset_index(drop=True)], axis=1
    )
    output["pair_weight_group_equal"] = 1.0 / pair_counts.to_numpy(dtype=float)
    # Move the weight back into the trace section if it was newly added.
    ordered = [column for column in TRACE_COLUMNS if column in output.columns]
    ordered += [column for column in LABEL_COLUMNS if column in output.columns]
    ordered += [column for column in AUDIT_COLUMNS if column in output.columns]
    ordered += MODEL_FEATURE_COLUMNS
    output = output.loc[:, list(dict.fromkeys(ordered))]
    validate_feature_table(output, expected_rows=len(frame))
    return output


def build_simple_pair_feature_table(pairs: pd.DataFrame) -> pd.DataFrame:
    """Build features and labels from the compact Chinese pair table.

    Labels use an absolute ``1e-4 S/cm`` unchanged band; conductivity
    columns remain audit/target fields only.
    """
    required = {
        "group_id", "pair_id", "化学式_a", "化学式_b", "电导率_a",
        "电导率_b", "电导率变化值", "温度", "合成方法", "family", "doi",
    }
    missing = sorted(required - set(pairs.columns))
    if missing:
        raise ValueError(f"Compact pair table is missing columns: {missing}")
    frame = pairs.copy()
    fa = frame["化学式_a"].astype(str).str.strip()
    fb = frame["化学式_b"].astype(str).str.strip()
    cache = _formula_descriptor_cache(pd.concat([fa, fb], ignore_index=True), show_progress=True)
    records = [
        _pair_numeric_features(cache[a], cache[b], a, b)
        for a, b in zip(fa, fb)
    ]
    numeric = pd.DataFrame.from_records(records, columns=MODEL_FEATURE_COLUMNS)
    def value(series: pd.Series) -> pd.Series:
        return pd.to_numeric(series.astype(str).str.replace(r"^[≈~]", "", regex=True), errors="raise")
    a = value(frame["电导率_a"])
    b = value(frame["电导率_b"])
    delta = b - a
    threshold = np.full(len(frame), TREND_ABSOLUTE_THRESHOLD_S_CM, dtype=float)
    labels = classify_trend_delta(delta)
    output = frame[["group_id", "pair_id", "化学式_a", "化学式_b", "电导率_a", "电导率_b", "电导率变化值", "温度", "合成方法", "family", "doi"]].copy()
    output["absolute_change"] = delta.abs()
    output["absolute_threshold"] = threshold
    output["log10_ratio_B_over_A"] = np.log10(b / a)
    output["trend_label"] = labels
    output["trend_class"] = pd.Series(labels).map({"increase": 1, "decrease": -1, "unchanged": 0}).to_numpy()
    output["pair_weight_group_equal"] = 1.0 / output.groupby("group_id")["pair_id"].transform("size")
    result = pd.concat(
        [output.reset_index(drop=True), numeric.reset_index(drop=True)], axis=1
    )
    validate_feature_table(result, expected_rows=len(frame))
    return result


def validate_feature_table(frame: pd.DataFrame, *, expected_rows: int | None = None) -> None:
    """Validate feature count, finiteness, symmetry fields, and group weights."""

    if expected_rows is not None and len(frame) != expected_rows:
        raise ValueError(
            f"Feature row count changed: expected {expected_rows}, got {len(frame)}."
        )
    missing = sorted(set(MODEL_FEATURE_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Feature table is missing model columns: {missing}")
    if len(MODEL_FEATURE_COLUMNS) != len(set(MODEL_FEATURE_COLUMNS)):
        raise ValueError("The model feature list must contain unique columns.")
    numeric = frame[MODEL_FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=float)
    if np.isinf(values).any():
        bad_columns = numeric.columns[np.isinf(values).any(axis=0)].tolist()
        raise ValueError(f"Infinite values found in model features: {bad_columns}")
    forbidden_missing = [
        column for column in numeric.columns
        if column not in OPTIONAL_MODEL_FEATURE_COLUMNS and numeric[column].isna().any()
    ]
    if forbidden_missing:
        raise ValueError(
            f"Unexpected missing values found in model features: {forbidden_missing}"
        )
    magnitude_pairs = {
        "abs_delta_entropy_c_l": "delta_entropy_c_l",
        "abs_delta_log10_rho_plus_incl_li": "delta_log10_rho_plus_incl_li",
        "abs_delta_log10_phi_plus_incl_li": "delta_log10_phi_plus_incl_li",
        "abs_delta_field_x_r_diff": "delta_field_x_r_diff",
        "abs_delta_n_Li": "delta_n_Li",
    }
    for magnitude, signed in magnitude_pairs.items():
        if magnitude in frame and signed in frame and not np.allclose(frame[magnitude], frame[signed].abs()):
            raise ValueError(f"{magnitude} is inconsistent with abs({signed}).")
    if not frame["composition_L1_distance"].between(0.0, 2.0 + 1e-9).all():
        raise ValueError("composition_L1_distance must be within [0, 2].")
    if "pair_id" in frame.columns and frame["pair_id"].duplicated().any():
        raise ValueError("pair_id must remain unique in the feature table.")
    if {"group_id", "pair_weight_group_equal"}.issubset(frame.columns):
        group_weight = frame.groupby("group_id")["pair_weight_group_equal"].sum()
        if not np.allclose(group_weight.to_numpy(dtype=float), 1.0):
            raise ValueError("pair_weight_group_equal must sum to 1 within each group.")


def build_prediction_features(
    formula_a: str,
    formula_b: str,
    family: str,
) -> pd.DataFrame:
    """Build one inference row from formulas and family only.

    The supplied order is preserved: every signed delta is B minus A.  The
    synthesis method is intentionally absent because the selected features
    are composition-only; callers may assume the same preparation method.
    """

    cache = _formula_descriptor_cache([formula_a, formula_b])
    features = _pair_numeric_features(
        cache[str(formula_a).strip()],
        cache[str(formula_b).strip()],
        str(formula_a).strip(),
        str(formula_b).strip(),
    )
    row = {
        "formula_a": str(formula_a).strip(),
        "formula_b": str(formula_b).strip(),
        "family": str(family).strip(),
        "family_normalized": normalize_family(family),
        "assumed_same_preparation": True,
        **features,
    }
    return pd.DataFrame([row], columns=[
        "formula_a",
        "formula_b",
        "family",
        "family_normalized",
        "assumed_same_preparation",
        *MODEL_FEATURE_COLUMNS,
    ])


def feature_schema() -> dict[str, object]:
    """Return a machine-readable schema for training and inference."""

    return {
        "schema_version": "trend_pair_features_v3_selected_absolute_a_b_delta",
        "data_version": "data-trend-v2",
        "model_feature_count": len(MODEL_FEATURE_COLUMNS),
        "model_feature_columns": MODEL_FEATURE_COLUMNS,
        "feature_groups": {
            "signed_delta": SIGNED_DELTA_FEATURES,
            "pair_mean": MEAN_FEATURES,
            "change_magnitude": MAGNITUDE_FEATURES,
            "absolute_a_baseline": A_BASELINE_FEATURES,
            "absolute_b_baseline": B_BASELINE_FEATURES,
        },
        "absolute_descriptor_selection": {
            "source": "consensus importance from abs_v0_f26 LightGBM, Random Forest, and NGBoost",
            "descriptors": SELECTED_ABSOLUTE_DESCRIPTORS,
            "representation": (
                "descriptor_A, descriptor_B, and descriptor_B - descriptor_A"
            ),
        },
        "signed_delta_definition": "descriptor_B - descriptor_A",
        "log_definition": {
            "rho_plus": "log10(rho_plus_incl_li)",
            "phi_plus": "log10(phi_plus_incl_li)",
            "log_rho_ratio": "existing main.features log10(clip(rho_plus/rho_minus, 1e-3)); then B-A",
        },
        "composition_L1_definition": "sum_e(abs(x_e_B - x_e_A)) using fractional atomic compositions",
        "prediction_inputs": ["formula_a", "formula_b", "family"],
        "preparation_assumption": (
            "same preparation for A and B; unknown values are retained under "
            "the user-specified default and remain audit metadata"
        ),
        "measurement_type_rule": (
            "total and bulk conductivity are separate pairing groups; type is metadata, not a model feature"
        ),
        "numeric_model_feature_columns": MODEL_FEATURE_COLUMNS,
        "categorical_model_feature_columns": ["family_normalized"],
        "model_input_columns": ["family_normalized", *MODEL_FEATURE_COLUMNS],
        "training_metadata_columns": TRACE_COLUMNS,
        "label_columns": LABEL_COLUMNS,
        "audit_columns": AUDIT_COLUMNS,
        "forbidden_model_columns": [
            *AUDIT_COLUMNS,
            *LABEL_COLUMNS,
            "pair_id",
            "group_id",
            "doi",
            "id_a",
            "id_b",
            "source_row_a",
            "source_row_b",
        ],
    }


def write_schema(
    feature_list_path: str | Path = DEFAULT_FEATURE_LIST_OUTPUT,
    schema_path: str | Path = DEFAULT_SCHEMA_OUTPUT,
) -> tuple[Path, Path]:
    feature_list_path = Path(feature_list_path)
    schema_path = Path(schema_path)
    feature_list_path.parent.mkdir(parents=True, exist_ok=True)
    schema_path.parent.mkdir(parents=True, exist_ok=True)
    feature_list_path.write_text(
        "\n".join(MODEL_FEATURE_COLUMNS) + "\n", encoding="utf-8"
    )
    schema_path.write_text(
        json.dumps(feature_schema(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return feature_list_path, schema_path


def write_pair_feature_file(
    input_path: str | Path = DEFAULT_PAIR_INPUT,
    output_path: str | Path = DEFAULT_TRAIN_OUTPUT,
    feature_list_path: str | Path = DEFAULT_FEATURE_LIST_OUTPUT,
    schema_path: str | Path = DEFAULT_SCHEMA_OUTPUT,
) -> dict[str, object]:
    """Read the pair table, compute features, and write the training file."""

    input_path = Path(input_path)
    output_path = Path(output_path)
    pairs = pd.read_csv(input_path, keep_default_na=False)
    output = build_pair_feature_table(pairs)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    feature_list_path, schema_path = write_schema(feature_list_path, schema_path)
    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "input_rows": int(len(pairs)),
        "output_rows": int(len(output)),
        "model_feature_count": len(MODEL_FEATURE_COLUMNS),
        "unique_formulas": int(
            pd.unique(pd.concat([pairs["formula_a"], pairs["formula_b"]])).size
        ),
        "family_context": "retained as categorical metadata; not counted among numeric features",
        "feature_list_path": str(feature_list_path),
        "schema_path": str(schema_path),
    }


def main() -> None:
    """Build trend pair features from the pairs table (point-run entry).

    Run directly via ``python main/trend/features.py`` (the "Run" button).
    Thin alias of ``features_simple``: reads the compact pairs table and emits
    the selected pair features consumed by the split -> train chain.
    Input   : data/trend/data-trend-v1-pairs.csv
    Output  : data/trend/data-trend-v2-pairs-feature.csv
    """
    pairs = pd.read_csv(PAIRS_INPUT, dtype=str, keep_default_na=False)
    output = build_simple_pair_feature_table(pairs)
    DEFAULT_PAIRS_FEATURE.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(DEFAULT_PAIRS_FEATURE, index=False)
    write_schema()
    print(f"pair_rows={len(output)} feature_columns={len(MODEL_FEATURE_COLUMNS)} "
          f"output={DEFAULT_PAIRS_FEATURE.resolve()}")


if __name__ == "__main__":
    main()
