"""Prepare leakage-safe same-study pairs for trend-v3 regression."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

if __package__ is None:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from pymatgen.core import Composition
from sklearn.model_selection import GroupShuffleSplit

from main.features import normalize_family
from main.paths import DATA_DIR, RUNS_DIR, portable_path
from main.trend.features import (
    A_BASELINE_FEATURES,
    MODEL_FEATURE_COLUMNS,
    _formula_descriptor_cache,
    _pair_numeric_features,
)


BASELINE_LOG10_COLUMN = "log10_conductivity_a"
TARGET_COLUMN = "log10_conductivity_b"
LOG_RATIO_COLUMN = "delta_log10_conductivity"
WEIGHT_COLUMN = "pair_weight_group_equal"
SPLIT_GROUP_COLUMN = "split_group"
DEFAULT_SOURCE = DATA_DIR / "absolute" / "data-absolute-v2.csv"
DEFAULT_RUN_NAME = "trend_reg_v3_conditional_f43_logsigmaB_l2_optuna50_seed42"
DEFAULT_RUN_DIR = RUNS_DIR / "trend" / DEFAULT_RUN_NAME

# The measured conductivity of A replaces the redundant A endpoint block. Each
# removed A descriptor can be reconstructed from its B value and directed delta.
MODEL_FEATURE_COLUMNS_V3 = [
    column for column in MODEL_FEATURE_COLUMNS if column not in A_BASELINE_FEATURES
] + [BASELINE_LOG10_COLUMN]


@dataclass(frozen=True)
class PairRegressionConfig:
    """Options for strict same-study pair preparation."""

    source_path: Path = DEFAULT_SOURCE
    min_conductivity: float = 1e-6
    room_temperature_min_c: float = 20.0
    room_temperature_max_c: float = 30.0
    exclude_ambiguous_multi_doi: bool = True
    exclude_close_matches: bool = True
    test_fraction: float = 0.15
    validation_fraction: float = 0.15
    seed: int = 42


class _DisjointSet:
    def __init__(self, values: list[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            parent = self.parent[value]
            self.parent[value] = root
            value = parent
        return root

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            keep, merge = sorted((left_root, right_root))
            self.parent[merge] = keep


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return portable_path(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _normalize_doi(value: object) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", text)
    return text.rstrip("./")


def _extract_note_field(note: object, field: str) -> str:
    match = re.search(rf"(?:^|;)\s*{re.escape(field)}=([^;]+)", str(note))
    if not match:
        return "unknown"
    value = match.group(1).strip().casefold()
    if value in {"", "unknown", "unknow", "na", "nan", "none"}:
        return "unknown"
    return re.sub(r"\s+", " ", value)


def _temperature_c(note: object) -> float | None:
    match = re.search(r"measured_temperature_C=([^;]+)", str(note), re.I)
    if not match:
        return None
    try:
        value = float(match.group(1).strip())
    except ValueError:
        return math.nan
    return value


def _temperature_basis(note: object) -> str:
    text = str(note)
    value = _temperature_c(text)
    if value is not None and np.isfinite(value):
        return "room_temperature_numeric"
    if re.search(r"measured_temperature=RT(?:;|$)", text, re.I):
        return "room_temperature_reported"
    if "lowest_extrapolation_temperature_K=" in text:
        return "room_temperature_extrapolated"
    return "room_temperature_unspecified"


def _measurement_basis(row: pd.Series) -> str:
    target = float(row["conductivity_S_cm-1"])
    for column, label in (("IC (Total)", "total"), ("IC (Bulk)", "bulk")):
        value = pd.to_numeric(row.get(column), errors="coerce")
        if np.isfinite(value) and np.isclose(float(value), target):
            return label
    note = str(row.get("note", "")).casefold()
    has_bulk = bool(re.search(r"\bbulk\b|σbulk", note))
    has_total = bool(re.search(r"\btotal\b|σtot", note))
    if has_bulk and not has_total:
        return "bulk"
    if has_total and not has_bulk:
        return "total"
    return "unknown"


def _phase_basis(note: object) -> str:
    explicit = _extract_note_field(note, "amorphous")
    if explicit != "unknown":
        return "amorphous" if explicit in {"1", "true", "yes"} else "crystalline"
    text = str(note).casefold()
    if "amorphous" in text and "crystalline" not in text:
        return "amorphous"
    if "crystalline" in text and "amorphous" not in text:
        return "crystalline"
    return "unknown"


def _source_kind(ref: object) -> str:
    text = str(ref).casefold()
    if "caltech" in text:
        return "caltech_extrapolated"
    if "liverpool" in text:
        return "liverpool_measured"
    if "literature_additions" in text:
        return "curated_literature"
    return "obelix_checked"


def _composition_info(formula: str) -> tuple[tuple[str, ...], tuple[float, ...]]:
    composition = Composition(formula).fractional_composition
    amounts = composition.get_el_amt_dict()
    elements = tuple(sorted(amounts))
    return elements, tuple(float(amounts[element]) for element in elements)


def _related_substitution(left: tuple[str, ...], right: tuple[str, ...]) -> bool:
    left_set, right_set = set(left), set(right)
    union = left_set | right_set
    intersection = left_set & right_set
    return (
        left_set != right_set
        and len(left_set ^ right_set) <= 2
        and len(intersection) >= 2
        and len(intersection) / len(union) >= 0.5
    )


def _temperature_pair_status(row_a: pd.Series, row_b: pd.Series) -> str | None:
    value_a = pd.to_numeric(row_a["temperature_C"], errors="coerce")
    value_b = pd.to_numeric(row_b["temperature_C"], errors="coerce")
    if np.isfinite(value_a) and np.isfinite(value_b):
        return "matched_numeric" if abs(float(value_a) - float(value_b)) <= 2.0 else None
    basis_a = str(row_a["temperature_basis"])
    basis_b = str(row_b["temperature_basis"])
    if "extrapolated" in basis_a or "extrapolated" in basis_b:
        return "matched_extrapolated" if basis_a == basis_b else None
    if basis_a == basis_b == "room_temperature_reported":
        return "matched_reported"
    return "room_temperature_incompletely_specified"


def clean_pair_source(
    config: PairRegressionConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Filter invalid, low-conductivity, and ambiguous source rows."""

    source = pd.read_csv(config.source_path)
    frame = source.copy()
    frame["conductivity_S_cm-1"] = pd.to_numeric(
        frame["Ionic conductivity (S cm-1)"], errors="coerce"
    )
    frame["doi_normalized"] = frame["DOI"].map(_normalize_doi)
    frame["family_normalized"] = frame["Family"].map(normalize_family)
    frame["formula"] = frame["True Composition"].astype(str).str.strip()
    frame["reduced_formula"] = frame["Reduced Composition"].astype(str).str.strip()
    frame["source_kind"] = frame["Ref"].map(_source_kind)
    frame["temperature_basis"] = frame["note"].map(_temperature_basis)
    frame["temperature_C"] = frame["note"].map(_temperature_c)
    frame["synthesis_basis"] = frame["note"].map(
        lambda value: _extract_note_field(value, "synthesis_method")
    )
    frame["phase_basis"] = frame["note"].map(_phase_basis)
    frame["measurement_basis"] = frame.apply(_measurement_basis, axis=1)
    composition_info = frame["formula"].map(_composition_info)
    frame["element_set"] = composition_info.map(lambda item: item[0])
    frame["composition_vector"] = composition_info.map(lambda item: item[1])

    conductivity = frame["conductivity_S_cm-1"]
    reasons = pd.Series("", index=frame.index, dtype=object)

    def mark(mask: pd.Series, reason: str) -> None:
        available = mask & reasons.eq("")
        reasons.loc[available] = reason

    mark(~np.isfinite(conductivity) | conductivity.le(0), "invalid conductivity")
    mark(conductivity.lt(config.min_conductivity), "below conductivity threshold")
    mark(frame["formula"].isin({"", "nan"}), "missing formula")
    mark(frame["doi_normalized"].isin({"", "nan"}), "missing DOI")
    mark(frame["family_normalized"].eq("unknown"), "unknown family")
    if config.exclude_ambiguous_multi_doi:
        mark(frame["doi_normalized"].str.contains("|", regex=False), "ambiguous multi-DOI record")
    if config.exclude_close_matches:
        mark(
            frame["note"].fillna("").str.contains(
                "close match: Yes", case=False, regex=False
            ),
            "close-match composition",
        )
    numeric_temperature = pd.to_numeric(frame["temperature_C"], errors="coerce")
    mark(
        numeric_temperature.notna()
        & ~numeric_temperature.between(
            config.room_temperature_min_c,
            config.room_temperature_max_c,
        ),
        "temperature outside room-temperature window",
    )

    excluded = frame.loc[reasons.ne("")].copy()
    excluded["excluded_reason"] = reasons.loc[reasons.ne("")]
    eligible = frame.loc[reasons.eq("")].copy().reset_index(drop=True)
    if eligible["ID"].astype(str).duplicated().any():
        raise ValueError("Eligible source IDs must be unique.")

    duplicate_keys = ["doi_normalized", "family_normalized", "reduced_formula"]
    duplicated = eligible.duplicated(duplicate_keys, keep=False)
    if duplicated.any():
        duplicate_rows = eligible.loc[duplicated].copy()
        duplicate_rows["excluded_reason"] = "duplicate formula within comparable group"
        excluded = pd.concat([excluded, duplicate_rows], ignore_index=True, sort=False)
        eligible = eligible.loc[~duplicated].reset_index(drop=True)

    summary = {
        "config": asdict(config),
        "input_rows": int(len(source)),
        "eligible_rows": int(len(eligible)),
        "excluded_rows": int(len(excluded)),
        "excluded_reasons": excluded["excluded_reason"].value_counts().to_dict(),
        "eligible_dois": int(eligible["doi_normalized"].nunique()),
        "eligible_families": int(eligible["family_normalized"].nunique()),
        "log10_conductivity": {
            "min": float(np.log10(eligible["conductivity_S_cm-1"]).min()),
            "median": float(np.log10(eligible["conductivity_S_cm-1"]).median()),
            "max": float(np.log10(eligible["conductivity_S_cm-1"]).max()),
        },
    }
    return eligible, excluded, summary


def _split_components(frame: pd.DataFrame) -> dict[str, str]:
    """Join DOI groups connected by a repeated formula."""

    dois = sorted(frame["doi_normalized"].astype(str).unique())
    disjoint = _DisjointSet(dois)
    for _, group in frame.groupby("reduced_formula", sort=True):
        group_dois = sorted(group["doi_normalized"].astype(str).unique())
        for other in group_dois[1:]:
            disjoint.union(group_dois[0], other)
    roots = {doi: disjoint.find(doi) for doi in dois}
    unique_roots = sorted(set(roots.values()))
    names = {root: f"component_{index:04d}" for index, root in enumerate(unique_roots, 1)}
    return {doi: names[root] for doi, root in roots.items()}


def build_same_study_pairs(eligible: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create all ordered-once pairs inside comparable DOI-family groups."""

    frame = eligible.copy()
    component_by_doi = _split_components(frame)
    group_columns = [
        "doi_normalized",
        "family_normalized",
        "measurement_basis",
        "synthesis_basis",
        "phase_basis",
    ]
    records: list[dict[str, Any]] = []
    group_count = 0
    pair_count = 0
    for keys, group in frame.groupby(group_columns, sort=True, dropna=False):
        if len(group) < 2:
            continue
        group = group.sort_values(["reduced_formula", "ID"], kind="stable")
        group_count += 1
        digest = hashlib.sha256("|".join(map(str, keys)).encode()).hexdigest()[:10]
        group_id = f"v3grp_{group_count:04d}_{digest}"
        rows = list(group.iterrows())
        candidates: dict[tuple[int, int], str] = {}
        for _, same_elements in group.groupby("element_set", sort=True):
            ordered = list(
                same_elements.sort_values(
                    ["composition_vector", "ID"], kind="stable"
                ).iterrows()
            )
            for (index_a, _), (index_b, _) in zip(ordered, ordered[1:]):
                candidates[tuple(sorted((index_a, index_b)))] = "adjacent_concentration"
        for (index_a, row_a), (index_b, row_b) in itertools.combinations(rows, 2):
            if _related_substitution(row_a["element_set"], row_b["element_set"]):
                candidates[tuple(sorted((index_a, index_b)))] = "related_element_substitution"
        for (index_a, index_b), pairing_strategy in sorted(candidates.items()):
            row_a = frame.loc[index_a]
            row_b = frame.loc[index_b]
            if row_a["reduced_formula"] == row_b["reduced_formula"]:
                continue
            temperature_status = _temperature_pair_status(row_a, row_b)
            if temperature_status is None:
                continue
            pair_count += 1
            sigma_a = float(row_a["conductivity_S_cm-1"])
            sigma_b = float(row_b["conductivity_S_cm-1"])
            log_sigma_a = math.log10(sigma_a)
            log_sigma_b = math.log10(sigma_b)
            delta_log = float(log_sigma_b - log_sigma_a)
            source_match = row_a["source_kind"] == row_b["source_kind"]
            measurement_known = row_a["measurement_basis"] != "unknown"
            confidence = (
                "high"
                if source_match
                and measurement_known
                and temperature_status != "room_temperature_incompletely_specified"
                else "moderate"
            )
            records.append({
                "pair_id": f"v3pair_{pair_count:06d}",
                "group_id": group_id,
                SPLIT_GROUP_COLUMN: component_by_doi[str(row_a["doi_normalized"])],
                "doi": row_a["doi_normalized"],
                "family": row_a["family_normalized"],
                "measurement_basis": row_a["measurement_basis"],
                "synthesis_basis": row_a["synthesis_basis"],
                "phase_basis": row_a["phase_basis"],
                "confidence_tier": confidence,
                "pairing_strategy": pairing_strategy,
                "temperature_pair_status": temperature_status,
                "id_a": str(row_a["ID"]),
                "id_b": str(row_b["ID"]),
                "formula_a": row_a["formula"],
                "formula_b": row_b["formula"],
                "reduced_formula_a": row_a["reduced_formula"],
                "reduced_formula_b": row_b["reduced_formula"],
                "source_kind_a": row_a["source_kind"],
                "source_kind_b": row_b["source_kind"],
                "temperature_basis_a": row_a["temperature_basis"],
                "temperature_basis_b": row_b["temperature_basis"],
                "conductivity_a_S_cm-1": sigma_a,
                "conductivity_b_S_cm-1": sigma_b,
                BASELINE_LOG10_COLUMN: log_sigma_a,
                TARGET_COLUMN: log_sigma_b,
                LOG_RATIO_COLUMN: delta_log,
                "conductivity_ratio_b_over_a": sigma_b / sigma_a,
                "delta_conductivity_S_cm-1": sigma_b - sigma_a,
            })
    pairs = pd.DataFrame.from_records(records)
    if pairs.empty:
        raise ValueError("No comparable same-study pairs were created.")
    pair_counts = pairs.groupby("group_id")["pair_id"].transform("size")
    pairs[WEIGHT_COLUMN] = 1.0 / pair_counts.to_numpy(dtype=float)
    summary = {
        "pair_rows": int(len(pairs)),
        "comparable_groups": int(pairs["group_id"].nunique()),
        "paired_dois": int(pairs["doi"].nunique()),
        "split_components": int(pairs[SPLIT_GROUP_COLUMN].nunique()),
        "paired_source_rows": int(
            len(set(pairs["id_a"]) | set(pairs["id_b"]))
        ),
        "confidence_tiers": pairs["confidence_tier"].value_counts().to_dict(),
        "family_pair_counts": pairs["family"].value_counts().to_dict(),
        "target": {
            "name": TARGET_COLUMN,
            "definition": "log10(sigma_B)",
            "min": float(pairs[TARGET_COLUMN].min()),
            "median": float(pairs[TARGET_COLUMN].median()),
            "max": float(pairs[TARGET_COLUMN].max()),
        },
        "derived_change": {
            "log_ratio": "log10(sigma_B) - log10(sigma_A)",
            "absolute_change": "sigma_B - sigma_A",
        },
    }
    return pairs, summary


def build_regression_feature_table(pairs: pd.DataFrame) -> pd.DataFrame:
    """Compute the existing F54 pair descriptors."""

    formulas = pd.concat([pairs["formula_a"], pairs["formula_b"]], ignore_index=True)
    cache = _formula_descriptor_cache(formulas, show_progress=True)
    feature_rows = []
    total = len(pairs)
    for index, row in enumerate(pairs.itertuples(index=False), start=1):
        formula_a = str(row.formula_a).strip()
        formula_b = str(row.formula_b).strip()
        feature_rows.append(
            _pair_numeric_features(
                cache[formula_a], cache[formula_b], formula_a, formula_b
            )
        )
        if index == total or index % 1000 == 0:
            print(f"Built pair features {index}/{total}", flush=True)
    features = pd.DataFrame.from_records(feature_rows, columns=MODEL_FEATURE_COLUMNS)
    result = pd.concat(
        [pairs.reset_index(drop=True), features.reset_index(drop=True)], axis=1
    )
    numeric = result[MODEL_FEATURE_COLUMNS_V3].apply(pd.to_numeric, errors="coerce")
    if np.isinf(numeric.to_numpy(dtype=float)).any():
        raise ValueError("Pair features contain infinite values.")
    required_finite = [
        column for column in MODEL_FEATURE_COLUMNS_V3 if not numeric[column].isna().any()
    ]
    if not required_finite:
        raise ValueError("No usable numeric pair features were created.")
    return result


def _split_once(
    frame: pd.DataFrame,
    *,
    held_out_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    best: tuple[float, np.ndarray, np.ndarray] | None = None
    target_rows = len(frame) * held_out_fraction
    change_mean = float(frame[LOG_RATIO_COLUMN].mean())
    change_std = float(frame[LOG_RATIO_COLUMN].std())
    for offset in range(500):
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=held_out_fraction,
            random_state=seed + offset,
        )
        fit_index, held_index = next(
            splitter.split(frame, groups=frame[SPLIT_GROUP_COLUMN].astype(str))
        )
        held = frame.iloc[held_index]
        row_deviation = abs(len(held) - target_rows) / len(frame)
        mean_deviation = abs(float(held[LOG_RATIO_COLUMN].mean()) - change_mean)
        std_deviation = abs(float(held[LOG_RATIO_COLUMN].std()) - change_std)
        score = row_deviation + 0.1 * mean_deviation + 0.1 * std_deviation
        if best is None or score < best[0]:
            best = (score, fit_index, held_index)
    if best is None:
        raise RuntimeError("No grouped split candidate was created.")
    return frame.iloc[best[1]].copy(), frame.iloc[best[2]].copy()


def split_pair_features(
    frame: pd.DataFrame,
    config: PairRegressionConfig,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Split connected DOI-formula components into train, validation, and test."""

    train_validation, test = _split_once(
        frame, held_out_fraction=config.test_fraction, seed=config.seed
    )
    relative_validation = config.validation_fraction / (1.0 - config.test_fraction)
    train, validation = _split_once(
        train_validation,
        held_out_fraction=relative_validation,
        seed=config.seed + 1,
    )
    splits = {
        "train": train.reset_index(drop=True),
        "validation": validation.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }
    for left, right in itertools.combinations(splits, 2):
        left_frame, right_frame = splits[left], splits[right]
        for column in (SPLIT_GROUP_COLUMN, "doi"):
            overlap = set(left_frame[column].astype(str)) & set(
                right_frame[column].astype(str)
            )
            if overlap:
                raise ValueError(f"{column} leakage between {left} and {right}.")
        left_formulas = set(left_frame["reduced_formula_a"]) | set(
            left_frame["reduced_formula_b"]
        )
        right_formulas = set(right_frame["reduced_formula_a"]) | set(
            right_frame["reduced_formula_b"]
        )
        if left_formulas & right_formulas:
            raise ValueError(f"Formula leakage between {left} and {right}.")
    summary = {
        "method": "GroupShuffleSplit over DOI-formula connected components",
        "selection_distribution": LOG_RATIO_COLUMN,
        "seed": config.seed,
        "fractions_requested": {
            "train": 1.0 - config.validation_fraction - config.test_fraction,
            "validation": config.validation_fraction,
            "test": config.test_fraction,
        },
        "splits": {
            name: {
                "pair_rows": int(len(part)),
                "fraction": float(len(part) / len(frame)),
                "dois": int(part["doi"].nunique()),
                "groups": int(part["group_id"].nunique()),
                "components": int(part[SPLIT_GROUP_COLUMN].nunique()),
                "log10_sigma_b_mean": float(part[TARGET_COLUMN].mean()),
                "log10_sigma_b_std": float(part[TARGET_COLUMN].std()),
                "delta_log10_ratio_mean": float(part[LOG_RATIO_COLUMN].mean()),
                "delta_log10_ratio_std": float(part[LOG_RATIO_COLUMN].std()),
            }
            for name, part in splits.items()
        },
        "leakage_checks": {
            "doi_overlap": 0,
            "reduced_formula_overlap": 0,
            "component_overlap": 0,
        },
    }
    return splits, summary


def prepare_regression_data(
    config: PairRegressionConfig,
    output_dir: Path,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Run cleaning, pairing, feature generation, and splitting."""

    output_dir.mkdir(parents=True, exist_ok=True)
    eligible, excluded, clean_summary = clean_pair_source(config)
    pairs, pair_summary = build_same_study_pairs(eligible)
    featured = build_regression_feature_table(pairs)
    splits, split_summary = split_pair_features(featured, config)

    eligible.to_csv(output_dir / "eligible_source_rows.csv", index=False)
    excluded.to_csv(output_dir / "excluded_source_rows.csv", index=False)
    featured.to_csv(output_dir / "all_pair_features.csv", index=False)
    for name, part in splits.items():
        part.to_csv(output_dir / f"{name}.csv", index=False)
    summary = {
        "source": portable_path(config.source_path),
        "cleaning": clean_summary,
        "pairing": pair_summary,
        "split": split_summary,
        "model_features": MODEL_FEATURE_COLUMNS_V3,
        "removed_redundant_features": A_BASELINE_FEATURES,
    }
    write_json(output_dir / "data_manifest.json", summary)
    return splits, summary
