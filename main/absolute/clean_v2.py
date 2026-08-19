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

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from main.features import normalize_family, parse_conductivity
from main.paths import DATA_DIR


ABSOLUTE_DIR = DATA_DIR / "absolute"

# Broad source labels that can be made formula-specific without changing phase.
FORMULA_FAMILY_RELABEL = {
    "LiNbCl4O": "oxyhalides",
    "LiTaCl4O": "oxyhalides",
}

# Restrict target selection when one formula is reported for distinct structures.
FORMULA_PREFERRED_FAMILY = {
    "Li1.6Al0.6Ge1.4P3O12": "nasicon",
    "Li10Sn(PS6)2": "lgps",
    "Li7P3S11": "thio_lisicon",
}


@dataclass
class CleanAbsoluteV2Config:
    """Options for producing a formula-unique, model-ready v2 table."""

    input_path: Path = ABSOLUTE_DIR / "data-absolute-v2.csv"
    output_path: Path = ABSOLUTE_DIR / "data-absolute-v2-model-clean.csv"
    excluded_path: Path = ABSOLUTE_DIR / "data-absolute-v2-model-clean-excluded.csv"
    formula_audit_path: Path = ABSOLUTE_DIR / "data-absolute-v2-model-clean-formula-audit.csv"
    quality_audit_path: Path = ABSOLUTE_DIR / "data-absolute-v2-model-clean-quality-audit.csv"
    summary_path: Path = ABSOLUTE_DIR / "data-absolute-v2-model-clean-summary.json"
    min_conductivity: float = 1e-6
    liverpool_temperature_min_c: float = 20.0
    liverpool_temperature_max_c: float = 30.0
    excluded_caltech_ids: tuple[str, ...] = (
        "caltech_icsd_65051",
        "caltech_icsd_100169",
    )


@dataclass
class CleanAbsoluteV2Result:
    table: pd.DataFrame
    excluded: pd.DataFrame
    formula_audit: pd.DataFrame
    quality_audit: pd.DataFrame
    summary: dict

    def to_files(self, config: CleanAbsoluteV2Config) -> None:
        config.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.table.to_csv(config.output_path, index=False)
        self.excluded.to_csv(config.excluded_path, index=False)
        self.formula_audit.to_csv(config.formula_audit_path, index=False)
        self.quality_audit.to_csv(config.quality_audit_path, index=False)
        config.summary_path.write_text(
            json.dumps(self.summary, indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )


def _source_name(ref: object) -> str:
    text = str(ref).lower()
    if "liverpool" in text:
        return "liverpool"
    if "caltech" in text:
        return "caltech"
    return "v1"


def _pipe(values: pd.Series) -> str:
    return "|".join(values.astype(str))


def clean_absolute_v2_data(
    config: CleanAbsoluteV2Config | None = None,
) -> CleanAbsoluteV2Result:
    """Filter the model threshold and retain one representative per formula."""

    config = config or CleanAbsoluteV2Config()
    source = pd.read_csv(config.input_path, dtype=str, keep_default_na=False)
    working = source.copy()
    parsed = working["Ionic conductivity (S cm-1)"].map(parse_conductivity)
    working["_conductivity"] = [value for value, _ in parsed]
    working["_qualifier"] = [qualifier for _, qualifier in parsed]
    working["_row_order"] = np.arange(len(working))
    working["_source"] = working["Ref"].map(_source_name)
    working["_source_rank"] = working["_source"].map(
        {"v1": 0, "liverpool": 1, "caltech": 2}
    )
    working["_checked_rank"] = ~working["Checked"].astype(str).str.lower().isin(
        {"1", "true", "yes"}
    )
    working["_log10"] = np.log10(
        pd.to_numeric(working["_conductivity"], errors="coerce").where(
            pd.to_numeric(working["_conductivity"], errors="coerce") > 0
        )
    )
    working["Family"] = working["Family"].map(normalize_family)
    for formula, family in FORMULA_FAMILY_RELABEL.items():
        working.loc[
            working["Reduced Composition"].eq(formula), "Family"
        ] = family
    working["_liverpool_temperature_c"] = pd.to_numeric(
        working["note"].str.extract(r"measured_temperature_C=([^;]+)")[0],
        errors="coerce",
    )

    conductivity = pd.to_numeric(working["_conductivity"], errors="coerce")
    invalid_mask = ~conductivity.gt(0)
    low_mask = conductivity.lt(config.min_conductivity) & ~invalid_mask
    liverpool_temperature_mask = working["_source"].eq("liverpool") & (
        working["_liverpool_temperature_c"].lt(
            config.liverpool_temperature_min_c
        )
        | working["_liverpool_temperature_c"].gt(
            config.liverpool_temperature_max_c
        )
        | working["_liverpool_temperature_c"].isna()
    )
    excluded_extrapolation_mask = (
        working["_source"].eq("caltech")
        & working["ID"].isin(config.excluded_caltech_ids)
    )
    initial_remove_mask = (
        invalid_mask
        | low_mask
        | liverpool_temperature_mask
        | excluded_extrapolation_mask
    )
    removed_initial = working[initial_remove_mask].copy()
    removed_initial["removed_reason"] = np.select(
        [
            invalid_mask[initial_remove_mask],
            low_mask[initial_remove_mask],
            liverpool_temperature_mask[initial_remove_mask],
            excluded_extrapolation_mask[initial_remove_mask],
        ],
        [
            "invalid or non-positive conductivity",
            f"model threshold: conductivity < {config.min_conductivity:g}",
            (
                "Liverpool temperature outside model range "
                f"[{config.liverpool_temperature_min_c:g}, "
                f"{config.liverpool_temperature_max_c:g}] C"
            ),
            "explicitly excluded Caltech high-temperature extrapolation record",
        ],
        default="excluded by model cleaning policy",
    )
    eligible = working[~initial_remove_mask].copy()

    kept_indices: list[int] = []
    duplicate_dropped_indices: list[int] = []
    formula_audit_rows = []
    quality_rows = []

    for formula, group in eligible.groupby("Reduced Composition", sort=True):
        if len(group) == 1:
            kept_indices.append(group.index[0])
            continue

        median_log10 = float(group["_log10"].median())
        preferred_family = FORMULA_PREFERRED_FAMILY.get(formula, "")
        selection_pool = group
        if preferred_family:
            preferred_rows = group[group["Family"].eq(preferred_family)]
            if not preferred_rows.empty:
                selection_pool = preferred_rows
        ranked = selection_pool.sort_values(
            ["_conductivity", "_checked_rank", "_source_rank", "_row_order"],
            ascending=[False, True, True, True],
            kind="stable",
        )
        selected = ranked.iloc[0]
        dropped = group.index.difference([selected.name]).tolist()
        kept_indices.append(selected.name)
        duplicate_dropped_indices.extend(dropped)

        families = sorted(set(group["Family"]))
        values = pd.to_numeric(group["_conductivity"], errors="coerce")
        log10_span = float(group["_log10"].max() - group["_log10"].min())
        formula_audit_rows.append({
            "Reduced Composition": formula,
            "candidate_rows": int(len(group)),
            "distinct_conductivities": int(values.nunique()),
            "log10_min": float(group["_log10"].min()),
            "log10_max": float(group["_log10"].max()),
            "log10_span": log10_span,
            "median_log10": median_log10,
            "IDs": _pipe(group["ID"]),
            "conductivities": _pipe(group["Ionic conductivity (S cm-1)"]),
            "DOIs": _pipe(group["DOI"]),
            "sources": _pipe(group["_source"]),
            "normalized_families": "|".join(families),
            "preferred_family": preferred_family,
            "kept_ID": selected["ID"],
            "kept_conductivity": float(selected["_conductivity"]),
            "kept_DOI": selected["DOI"],
            "kept_source": selected["_source"],
            "kept_family": selected["Family"],
            "selection_rule": (
                f"highest reported conductivity within {preferred_family}"
                if preferred_family
                else "highest reported conductivity"
            ),
            "dropped_rows": int(len(dropped)),
        })
        quality_rows.append({
            "issue_type": "same_formula_multiple_records",
            "severity": "resolved",
            "Reduced Composition": formula,
            "IDs": _pipe(group["ID"]),
            "details": (
                f"{len(group)} records; log10 span={log10_span:.3f}; "
                f"kept {selected['ID']}"
            ),
        })
        if len(families) > 1:
            quality_rows.append({
                "issue_type": "conflicting_normalized_family",
                "severity": (
                    "resolved_by_preferred_family"
                    if preferred_family
                    else "manual_review"
                ),
                "Reduced Composition": formula,
                "IDs": _pipe(group["ID"]),
                "details": (
                    "|".join(families)
                    + (
                        f"; selected_family={preferred_family}"
                        if preferred_family
                        else ""
                    )
                ),
            })

    kept = eligible.loc[sorted(kept_indices)].copy()
    duplicate_dropped = eligible.loc[sorted(duplicate_dropped_indices)].copy()
    if not duplicate_dropped.empty:
        kept_id_by_formula = kept.set_index("Reduced Composition")["ID"].to_dict()
        duplicate_dropped["removed_reason"] = duplicate_dropped[
            "Reduced Composition"
        ].map(
            lambda formula: (
                "same formula representative selection; kept "
                f"{kept_id_by_formula[formula]} with highest reported conductivity"
            )
        )

    internal_columns = [column for column in kept.columns if column.startswith("_")]
    output = kept.sort_values("_row_order", kind="stable").drop(
        columns=internal_columns
    ).reset_index(drop=True)
    excluded = pd.concat(
        [removed_initial, duplicate_dropped], ignore_index=True, sort=False
    )
    excluded = excluded.drop(
        columns=[column for column in excluded.columns if column.startswith("_")]
    ).reset_index(drop=True)
    formula_audit = pd.DataFrame(formula_audit_rows).sort_values(
        ["log10_span", "candidate_rows"],
        ascending=[False, False],
        ignore_index=True,
    )
    quality_audit = pd.DataFrame(quality_rows, columns=[
        "issue_type",
        "severity",
        "Reduced Composition",
        "IDs",
        "details",
    ])

    summary = {
        "config": asdict(config),
        "input_rows": int(len(source)),
        "eligible_rows_after_threshold_and_temperature": int(len(eligible)),
        "output_rows": int(len(output)),
        "unique_output_formulas": int(output["Reduced Composition"].nunique()),
        "invalid_or_nonpositive_rows": int(invalid_mask.sum()),
        "below_conductivity_threshold_rows": int(low_mask.sum()),
        "liverpool_temperature_rows_removed": int(
            liverpool_temperature_mask.sum()
        ),
        "initial_policy_rows_removed_after_overlap": int(len(removed_initial)),
        "duplicate_formula_groups_resolved": int(len(formula_audit)),
        "duplicate_formula_rows_removed": int(len(duplicate_dropped)),
        "normalized_family_conflict_groups": int(
            quality_audit["issue_type"].eq("conflicting_normalized_family").sum()
        ),
        "preferred_family_selection_groups": int(
            len(FORMULA_PREFERRED_FAMILY)
        ),
        "formula_family_relabel_groups": int(len(FORMULA_FAMILY_RELABEL)),
        "high_temperature_extrapolation_rows_removed": int(
            excluded_extrapolation_mask.sum()
        ),
        "remaining_duplicate_formula_rows": int(
            output["Reduced Composition"].duplicated().sum()
        ),
        "selection_policy": (
            "one row per normalized reduced formula; highest reported "
            "conductivity within the preferred structural family when defined, "
            "otherwise highest overall; ties prefer checked v1, explicit "
            "room-temperature Liverpool, then Caltech"
        ),
    }
    return CleanAbsoluteV2Result(
        table=output,
        excluded=excluded,
        formula_audit=formula_audit,
        quality_audit=quality_audit,
        summary=summary,
    )


def main() -> None:
    config = CleanAbsoluteV2Config()
    result = clean_absolute_v2_data(config)
    result.to_files(config)
    print(json.dumps(result.summary, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
