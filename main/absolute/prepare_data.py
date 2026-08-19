from __future__ import annotations

import argparse
import os
import re
import sys

# Allow running this module directly with ``python main/absolute/prepare_data.py``.
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
from pymatgen.core import Composition

from main.features import normalize_family, parse_conductivity
from main.paths import DATA_DIR


ABSOLUTE_DIR = DATA_DIR / "absolute"
DEFAULT_INPUT = DATA_DIR / "database" / "obelix" / "processed" / "all-v1.csv"
DEFAULT_OUTPUT = ABSOLUTE_DIR / "data-absolute-v1.csv"
DEFAULT_EXCLUDED = ABSOLUTE_DIR / "data-absolute-v1-excluded.csv"
DEFAULT_AUDIT = ABSOLUTE_DIR / "data-absolute-v1-duplicate-audit.csv"
DEFAULT_V2_CALTECH_INPUT = (
    DATA_DIR / "database" / "caltech" / "ionic_conductivity_database .csv"
)
DEFAULT_V2_LIVERPOOL_INPUT = (
    DATA_DIR / "database" / "liverpool" / "LiIonDatabase.csv"
)
DEFAULT_V2_OUTPUT = ABSOLUTE_DIR / "data-absolute-v2.csv"
DEFAULT_V2_EXCLUDED = ABSOLUTE_DIR / "data-absolute-v2-excluded.csv"
DEFAULT_V2_AUDIT = ABSOLUTE_DIR / "data-absolute-v2-duplicate-audit.csv"
DEFAULT_V2_FAMILY_AUDIT = ABSOLUTE_DIR / "data-absolute-v2-family-audit.csv"
DEFAULT_V2_FAMILY_COUNTS = ABSOLUTE_DIR / "data-absolute-v2-family-counts.csv"

ABSOLUTE_COLUMNS = [
    "ID",
    "Reduced Composition",
    "Z",
    "True Composition",
    "Ionic conductivity (S cm-1)",
    "IC (Total)",
    "IC (Bulk)",
    "Space group",
    "Space group #",
    "a",
    "b",
    "c",
    "alpha",
    "beta",
    "gamma",
    "Family",
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
]


@dataclass
class PrepareAbsoluteConfig:
    """Options for building the absolute-model source table from all-v1."""

    input_path: Path = DEFAULT_INPUT
    output_path: Path = DEFAULT_OUTPUT
    excluded_path: Path = DEFAULT_EXCLUDED
    audit_path: Path = DEFAULT_AUDIT
    duplicate_log10_span_limit: float = 0.5
    exclude_conflicting_duplicate_formulas: bool = True


@dataclass
class PrepareAbsoluteV2Config:
    """Options for merging v1, Caltech, and Liverpool room-temperature data."""

    v1_input_path: Path = DEFAULT_OUTPUT
    caltech_input_path: Path = DEFAULT_V2_CALTECH_INPUT
    liverpool_input_path: Path = DEFAULT_V2_LIVERPOOL_INPUT
    output_path: Path = DEFAULT_V2_OUTPUT
    excluded_path: Path = DEFAULT_V2_EXCLUDED
    audit_path: Path = DEFAULT_V2_AUDIT
    family_audit_path: Path = DEFAULT_V2_FAMILY_AUDIT
    family_counts_path: Path = DEFAULT_V2_FAMILY_COUNTS
    liverpool_room_temperature_c: float = 25.0
    liverpool_temperature_tolerance_c: float = 5.0
    excluded_caltech_icsd_ids: tuple[str, ...] = ("65051", "100169")
    formula_log10_span_limit: float = 1.0
    keep_highest_for_conflicting_formulas: bool = True


@dataclass
class PrepareAbsoluteResult:
    table: pd.DataFrame
    excluded: pd.DataFrame
    duplicate_audit: pd.DataFrame
    summary: dict
    family_audit: pd.DataFrame | None = None
    family_counts: pd.DataFrame | None = None

    def to_files(self, config: PrepareAbsoluteConfig) -> None:
        config.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.table.to_csv(config.output_path, index=False)
        self.excluded.to_csv(config.excluded_path, index=False)
        self.duplicate_audit.to_csv(config.audit_path, index=False)
        if self.family_audit is not None and hasattr(config, "family_audit_path"):
            self.family_audit.to_csv(config.family_audit_path, index=False)
        if self.family_counts is not None and hasattr(config, "family_counts_path"):
            self.family_counts.to_csv(config.family_counts_path, index=False)


def _composition_or_error(formula: object) -> tuple[Composition | None, str]:
    text = str(formula).strip()
    if not text:
        return None, "missing formula"
    try:
        return Composition(text), ""
    except Exception as exc:
        return None, str(exc)


def _normalized_formula(formula: object) -> tuple[str, str, str]:
    composition, error = _composition_or_error(formula)
    if composition is None:
        return "", "", error
    return composition.reduced_formula, composition.formula, ""


def _source_note(row: pd.Series) -> str:
    parts = []
    note = str(row.get("备注", "")).strip()
    if note:
        parts.append(note)
    metadata = {
        "synthesis_method": row.get("化合物制备方法", ""),
        "measured_temperature": row.get("电导率测得温度（°C）", ""),
        "amorphous": row.get("非晶", ""),
        "source_year": row.get("文献年份", ""),
    }
    for key, value in metadata.items():
        text = str(value).strip()
        if text:
            parts.append(f"{key}={text}")
    return "; ".join(parts)


def _build_absolute_rows(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in frame.iterrows():
        reduced, true_formula, formula_error = _normalized_formula(row.get("化学式", ""))
        value, qualifier = parse_conductivity(row.get("电导率（室温，S/cm）", ""))
        output = {column: "" for column in ABSOLUTE_COLUMNS}
        output.update({
            "ID": row.get("ID", ""),
            "Reduced Composition": reduced,
            "True Composition": true_formula,
            "Ionic conductivity (S cm-1)": row.get("电导率（室温，S/cm）", ""),
            "Family": row.get("family", ""),
            "DOI": row.get("对应文献DOI", ""),
            "Checked": "1",
            "Ref": "data/database/obelix/processed/all-v1.csv",
            "note": _source_note(row),
        })
        output["_source_formula"] = row.get("化学式", "")
        output["_formula_error"] = formula_error
        output["_conductivity_value"] = value
        output["_conductivity_qualifier"] = qualifier
        rows.append(output)
    return pd.DataFrame(rows)


def _duplicate_audit(frame: pd.DataFrame) -> pd.DataFrame:
    valid = frame[
        frame["_formula_error"].eq("")
        & pd.to_numeric(frame["_conductivity_value"], errors="coerce").gt(0)
    ].copy()
    if valid.empty:
        return pd.DataFrame(columns=[
            "Reduced Composition",
            "row_count",
            "log10_min",
            "log10_max",
            "log10_span",
            "IDs",
            "conductivities",
            "DOIs",
        ])

    valid["_log10_conductivity"] = np.log10(
        pd.to_numeric(valid["_conductivity_value"], errors="coerce")
    )
    grouped = valid.groupby("Reduced Composition", sort=True, dropna=False)
    records = []
    for formula, group in grouped:
        if len(group) <= 1:
            continue
        records.append({
            "Reduced Composition": formula,
            "row_count": int(len(group)),
            "log10_min": float(group["_log10_conductivity"].min()),
            "log10_max": float(group["_log10_conductivity"].max()),
            "log10_span": float(
                group["_log10_conductivity"].max()
                - group["_log10_conductivity"].min()
            ),
            "IDs": "|".join(group["ID"].astype(str)),
            "conductivities": "|".join(group["Ionic conductivity (S cm-1)"].astype(str)),
            "DOIs": "|".join(group["DOI"].astype(str)),
        })
    return pd.DataFrame(records).sort_values(
        ["log10_span", "row_count"],
        ascending=[False, False],
        ignore_index=True,
    )


def prepare_absolute_data(
    config: PrepareAbsoluteConfig | None = None,
) -> PrepareAbsoluteResult:
    """Build a reproducible absolute-model source table from the latest all-v1."""

    config = config or PrepareAbsoluteConfig()
    source = pd.read_csv(config.input_path, dtype=str, keep_default_na=False)
    working = _build_absolute_rows(source)
    removed_frames = []

    formula_error = working["_formula_error"].ne("")
    if formula_error.any():
        removed = working[formula_error].copy()
        removed["removed_reason"] = "unparseable formula: " + removed["_formula_error"]
        removed_frames.append(removed)
    working = working[~formula_error].copy()

    missing_conductivity = ~pd.to_numeric(
        working["_conductivity_value"],
        errors="coerce",
    ).gt(0)
    if missing_conductivity.any():
        removed = working[missing_conductivity].copy()
        removed["removed_reason"] = (
            "missing or unparseable room-temperature conductivity"
        )
        removed_frames.append(removed)
    working = working[~missing_conductivity].copy()

    audit = _duplicate_audit(working)
    if config.exclude_conflicting_duplicate_formulas and not audit.empty:
        conflict_formulas = set(
            audit[
                audit["log10_span"] > config.duplicate_log10_span_limit
            ]["Reduced Composition"].astype(str)
        )
        conflict = working["Reduced Composition"].astype(str).isin(conflict_formulas)
        if conflict.any():
            removed = working[conflict].copy()
            span_by_formula = audit.set_index("Reduced Composition")["log10_span"]
            removed["removed_reason"] = (
                "conflicting duplicate formula: log10 conductivity span "
                + removed["Reduced Composition"].map(span_by_formula).round(3).astype(str)
                + f" > {config.duplicate_log10_span_limit:g}"
            )
            removed_frames.append(removed)
        working = working[~conflict].copy()

    excluded = (
        pd.concat(removed_frames, ignore_index=True)
        if removed_frames
        else pd.DataFrame(columns=[*working.columns, "removed_reason"])
    )
    output = working[ABSOLUTE_COLUMNS].reset_index(drop=True)
    excluded_output = excluded[[*ABSOLUTE_COLUMNS, "removed_reason"]].reset_index(drop=True)
    summary = {
        "config": asdict(config),
        "source_rows": int(len(source)),
        "output_rows": int(len(output)),
        "excluded_rows": int(len(excluded_output)),
        "duplicate_formula_groups": int(len(audit)),
        "conflicting_duplicate_formula_groups": int(
            (audit["log10_span"] > config.duplicate_log10_span_limit).sum()
            if not audit.empty else 0
        ),
    }
    return PrepareAbsoluteResult(
        table=output,
        excluded=excluded_output,
        duplicate_audit=audit,
        summary=summary,
    )


_DOI_URL_PREFIX = re.compile(
    r"^(?:https?://(?:dx\.)?doi\.org/|doi\s*:\s*)",
    flags=re.IGNORECASE,
)
_DOI_PATTERN = re.compile(r"^10\.\d{4,9}/\S+$", flags=re.IGNORECASE)
_SOURCE_PRIORITY = {"v1": 0, "caltech": 1, "liverpool": 2}


def _doi_tokens(value: object) -> tuple[str, ...]:
    """Normalize one or more pipe-separated DOI values."""
    tokens = []
    for raw_token in str(value).split("|"):
        token = _DOI_URL_PREFIX.sub("", raw_token.strip()).rstrip(".,").lower()
        if token and _DOI_PATTERN.fullmatch(token):
            tokens.append(token)
    return tuple(sorted(set(tokens)))


def _candidate_row(
    source_name: str,
    source_row: int,
    source_formula: object,
    source_doi: object,
    source_conductivity: object,
) -> dict:
    row = {column: "" for column in ABSOLUTE_COLUMNS}
    row.update({
        "_source_name": source_name,
        "_source_row": int(source_row),
        "_source_formula": source_formula,
        "_source_doi": source_doi,
        "_source_conductivity": source_conductivity,
        "_temperature_c": np.nan,
    })
    return row


def _v1_candidates(path: Path) -> pd.DataFrame:
    source = pd.read_csv(path, dtype=str, keep_default_na=False)
    rows = []
    for source_row, (_, input_row) in enumerate(source.iterrows(), start=1):
        row = _candidate_row(
            "v1",
            source_row,
            input_row.get("True Composition") or input_row.get("Reduced Composition"),
            input_row.get("DOI", ""),
            input_row.get("Ionic conductivity (S cm-1)", ""),
        )
        row.update({column: input_row.get(column, "") for column in ABSOLUTE_COLUMNS})
        rows.append(row)
    return pd.DataFrame(rows)


def _caltech_candidates(path: Path) -> pd.DataFrame:
    source = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    rows = []
    for source_row, (_, input_row) in enumerate(source.iterrows(), start=1):
        code = str(input_row.get("icsd_collectioncode", "")).strip()
        row = _candidate_row(
            "caltech",
            source_row,
            input_row.get("compound", ""),
            input_row.get("conductivity_doi", ""),
            input_row.get("conductivity_siemens_per_cm", ""),
        )
        row.update({
            "ID": f"caltech_icsd_{code or source_row}",
            "Ionic conductivity (S cm-1)": input_row.get(
                "conductivity_siemens_per_cm", ""
            ),
            "Checked": "0",
            "Ref": "data/database/caltech/ionic_conductivity_database .csv",
            "ICSD ID": code,
            "note": (
                f"lowest_extrapolation_temperature_K="
                f"{input_row.get('lowest_extrapolation_temperature_K', '')}"
            ),
            "_lowest_extrapolation_temperature_k": pd.to_numeric(
                input_row.get("lowest_extrapolation_temperature_K", ""),
                errors="coerce",
            ),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def _read_liverpool(path: Path) -> pd.DataFrame:
    source = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    if "composition" not in source.columns:
        source = pd.read_csv(
            path,
            sep="\t",
            skiprows=1,
            dtype=str,
            keep_default_na=False,
        )
    return source.loc[:, ~source.columns.astype(str).str.match(r"^Unnamed")]


def _liverpool_candidates(path: Path) -> pd.DataFrame:
    source = _read_liverpool(path)
    rows = []
    for source_row, (_, input_row) in enumerate(source.iterrows(), start=1):
        source_id = str(input_row.get("ID", "")).strip()
        temperature = pd.to_numeric(input_row.get("temperature", ""), errors="coerce")
        row = _candidate_row(
            "liverpool",
            source_row,
            input_row.get("composition", ""),
            input_row.get("source", ""),
            input_row.get("target", ""),
        )
        row.update({
            "ID": f"liverpool_{source_id or 'row'}_{source_row:04d}",
            "Ionic conductivity (S cm-1)": input_row.get("target", ""),
            "Family": input_row.get("family", ""),
            "Checked": "0",
            "Ref": "data/database/liverpool/LiIonDatabase.csv",
            "note": (
                f"source_id={source_id}; measured_temperature_C="
                f"{input_row.get('temperature', '')}; chemical_family="
                f"{input_row.get('ChemicalFamily', '')}"
            ),
            "_temperature_c": temperature,
        })
        rows.append(row)
    return pd.DataFrame(rows)


def _validate_v2_candidates(
    candidates: pd.DataFrame,
    config: PrepareAbsoluteV2Config,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid_rows = []
    excluded_rows = []
    for _, candidate in candidates.iterrows():
        row = candidate.to_dict()
        reduced, true_formula, formula_error = _normalized_formula(
            row.get("_source_formula", "")
        )
        doi_tokens = _doi_tokens(row.get("_source_doi", ""))
        conductivity, qualifier = parse_conductivity(
            row.get("_source_conductivity", "")
        )
        reason = ""
        if formula_error:
            reason = f"unparseable formula: {formula_error}"
        elif not doi_tokens:
            reason = "missing or invalid DOI"
        elif not np.isfinite(conductivity) or conductivity <= 0:
            reason = "missing, non-positive, or unparseable conductivity"
        elif row.get("_source_name") == "liverpool":
            temperature = pd.to_numeric(row.get("_temperature_c"), errors="coerce")
            lower = (
                config.liverpool_room_temperature_c
                - config.liverpool_temperature_tolerance_c
            )
            upper = (
                config.liverpool_room_temperature_c
                + config.liverpool_temperature_tolerance_c
            )
            if not np.isfinite(temperature) or not lower <= temperature <= upper:
                reason = f"Liverpool temperature outside room-temperature range [{lower:g}, {upper:g}] C"
        elif row.get("_source_name") == "caltech":
            icsd_id = str(row.get("ICSD ID", "")).strip()
            if icsd_id in config.excluded_caltech_icsd_ids:
                lowest_temperature = pd.to_numeric(
                    row.get("_lowest_extrapolation_temperature_k"), errors="coerce"
                )
                temperature_text = (
                    f"{lowest_temperature:g} K"
                    if np.isfinite(lowest_temperature)
                    else "unknown temperature"
                )
                reason = (
                    "explicitly excluded Caltech high-temperature extrapolation: "
                    f"ICSD {icsd_id}, {temperature_text}"
                )

        row["Reduced Composition"] = reduced
        row["True Composition"] = true_formula
        row["DOI"] = "|".join(doi_tokens)
        row["Ionic conductivity (S cm-1)"] = conductivity
        row["_doi_tokens"] = doi_tokens
        row["_conductivity_value"] = conductivity
        row["_conductivity_qualifier"] = qualifier
        if reason:
            row["removed_reason"] = reason
            excluded_rows.append(row)
        else:
            valid_rows.append(row)

    valid = pd.DataFrame(valid_rows)
    excluded = pd.DataFrame(excluded_rows)
    return valid, excluded


def _fill_caltech_families(candidates: pd.DataFrame) -> pd.DataFrame:
    candidates = candidates.copy()
    family_by_key: dict[tuple[str, str], str] = {}
    ordered = candidates.sort_values(
        "_source_name",
        key=lambda values: values.map(_SOURCE_PRIORITY),
        kind="stable",
    )
    for _, row in ordered.iterrows():
        family = normalize_family(row.get("Family", ""))
        if family in {"", "unknown", "unknow"}:
            continue
        for doi in row["_doi_tokens"]:
            family_by_key.setdefault((row["Reduced Composition"], doi), family)

    for index, row in candidates[candidates["_source_name"].eq("caltech")].iterrows():
        if normalize_family(row.get("Family", "")) not in {"", "unknown", "unknow"}:
            continue
        matches = [
            family_by_key.get((row["Reduced Composition"], doi))
            for doi in row["_doi_tokens"]
        ]
        matches = [family for family in matches if family]
        candidates.at[index, "Family"] = matches[0] if matches else "unknown"
    return candidates


def _specific_family_candidates(values: set[str]) -> set[str]:
    """Ignore a generic `other` label when a specific family is available."""
    normalized = {normalize_family(value) for value in values}
    normalized.discard("unknown")
    if len(normalized) > 1:
        normalized.discard("other")
    return normalized


def _infer_family_from_composition(formula: object) -> tuple[str, str]:
    """Assign a broad existing taxonomy label from composition only."""
    composition, error = _composition_or_error(formula)
    if composition is None:
        return "other", f"composition fallback: {error}"

    amounts = {element.symbol: float(amount) for element, amount in composition.items()}
    symbols = set(amounts)
    halides = symbols & {"F", "Cl", "Br", "I"}
    chalcogens = symbols & {"S", "Se", "Te"}
    has_oxygen = "O" in symbols

    if has_oxygen and halides:
        if "H" in symbols and "Li" in symbols:
            return "antiperovskite", "composition: Li-H-O-halide"
        if "B" in symbols and "F" in symbols:
            return "fluorooxoborate", "composition: B-O-F"
        if "P" in symbols and "F" in symbols:
            return "fluorophosphates", "composition: P-O-F"
        return "oxyhalides", "composition: oxide + halide"

    if chalcogens and halides:
        if "P" in symbols:
            return "argyrodites", "composition: Li-P-chalcogen-halide"
        return "halides", "composition: chalcogen-halide"

    if halides:
        return "halides", "composition: halide"

    if "H" in symbols and "Li" in symbols:
        return "hydrides", "composition: lithium hydride"

    if "N" in symbols and not has_oxygen:
        if "Si" in symbols:
            return "nitridosilicates", "composition: nitride + silicon"
        return "nitrides", "composition: nitride"

    if chalcogens and not has_oxygen:
        if "P" in symbols:
            network_formers = symbols & {"Si", "Ge", "Sn"}
            if network_formers:
                if amounts.get("Li", 0.0) >= 6 or amounts.get("S", 0.0) >= 8:
                    return "lgps", "composition: Li-M-P-S LGPS-like"
                return "thio_lisicon", "composition: Li-M-P-S"
            return "thio_phosphate", "composition: Li-P-chalcogen"
        if symbols & {"Si", "Ge", "Sn"} and amounts.get("Li", 0.0) >= 3:
            return (
                "chalcogenidotetrelates",
                "composition: lithium tetrel chalcogenide",
            )
        return "sulfides", "composition: chalcogenide"

    if has_oxygen:
        if "P" in symbols:
            if "B" in symbols:
                return "borophosphates", "composition: B-P-O"
            if "F" in symbols:
                return "fluorophosphates", "composition: P-O-F"
            if (
                amounts.get("P", 0.0) >= 2.5
                and amounts.get("O", 0.0) >= 10
                and symbols & {"Ti", "Zr", "Ge", "Sn", "Hf", "V", "Nb"}
            ):
                return "nasicon", "composition: M-P3-O12 NASICON-like"
            if symbols & {"Si", "Ge"}:
                return "lisicon", "composition: Li-(Si/Ge)-P-O"
            return "phosphates", "composition: phosphate"
        if "B" in symbols:
            return "borates", "composition: borate"
        if "Mo" in symbols:
            return "molybdates", "composition: molybdate"
        if (
            "Li" in symbols
            and "La" in symbols
            and amounts.get("O", 0.0) >= 10
            and amounts.get("Li", 0.0) >= 5
        ):
            return "garnet", "composition: Li-La-O12 garnet-like"
        cation_amount = sum(
            amount for symbol, amount in amounts.items() if symbol != "O"
        )
        if (
            2.5 <= amounts.get("O", 0.0) <= 3.5
            and 1.5 <= cation_amount <= 2.5
        ):
            return "perovskites", "composition: ABO3 stoichiometry"
        if "Si" in symbols:
            return "silicates", "composition: silicate"
        if "Ge" in symbols:
            return "germanates", "composition: germanate"
        return "oxides", "composition: oxide"

    return "other", "composition: taxonomy fallback"


def _label_unknown_families(
    candidates: pd.DataFrame,
    references: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Label unknown families using formula, DOI-series, then composition evidence."""
    candidates = candidates.copy()
    known = references[
        ~references["Family"].map(normalize_family).eq("unknown")
    ].copy()
    known["_family_normalized"] = known["Family"].map(normalize_family)

    formula_families = known.groupby("Reduced Composition")["_family_normalized"].agg(
        lambda values: set(values)
    ).to_dict()
    doi_families: dict[str, set[str]] = {}
    for _, row in known.iterrows():
        for doi in row["_doi_tokens"]:
            doi_families.setdefault(doi, set()).add(row["_family_normalized"])

    audit_rows = []
    unknown_mask = candidates["Family"].map(normalize_family).eq("unknown")
    for index, row in candidates[unknown_mask].iterrows():
        formula_options = _specific_family_candidates(
            formula_families.get(row["Reduced Composition"], set())
        )
        doi_options = _specific_family_candidates(set().union(*(
            doi_families.get(doi, set()) for doi in row["_doi_tokens"]
        )))

        if len(formula_options) == 1:
            family = next(iter(formula_options))
            confidence = "high"
            method = "exact_formula_unique_family"
            evidence = "|".join(sorted(formula_options))
        elif len(doi_options) == 1:
            family = next(iter(doi_options))
            confidence = "medium"
            method = "doi_series_unique_family"
            evidence = "|".join(sorted(doi_options))
        else:
            family, evidence = _infer_family_from_composition(
                row["True Composition"]
            )
            confidence = "low"
            method = "composition_rule"
            conflicts = sorted(formula_options | doi_options)
            if conflicts:
                evidence += "; conflicting_reference_labels=" + "|".join(conflicts)

        candidates.at[index, "Family"] = family
        audit_rows.append({
            "ID": row["ID"],
            "Reduced Composition": row["Reduced Composition"],
            "DOI": row["DOI"],
            "source": row["_source_name"],
            "previous_family": row["Family"],
            "recommended_family": family,
            "confidence": confidence,
            "assignment_method": method,
            "evidence": evidence,
        })

    audit = pd.DataFrame(audit_rows, columns=[
        "ID",
        "Reduced Composition",
        "DOI",
        "source",
        "previous_family",
        "recommended_family",
        "confidence",
        "assignment_method",
        "evidence",
    ])
    normalized = candidates["Family"].map(normalize_family)
    counts = (
        normalized.value_counts()
        .rename_axis("family")
        .reset_index(name="row_count")
    )
    counts["fraction"] = counts["row_count"] / len(candidates)
    return candidates, audit, counts


def _duplicate_components(candidates: pd.DataFrame) -> pd.Series:
    """Group rows sharing a normalized formula and at least one DOI."""
    parent = list(range(len(candidates)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    first_by_key: dict[tuple[str, str], int] = {}
    for index, row in candidates.reset_index(drop=True).iterrows():
        for doi in row["_doi_tokens"]:
            key = (row["Reduced Composition"], doi)
            if key in first_by_key:
                union(first_by_key[key], index)
            else:
                first_by_key[key] = index
    return pd.Series([find(index) for index in range(len(candidates))])


def _deduplicate_v2_candidates(
    candidates: pd.DataFrame,
    room_temperature_c: float = 25.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    candidates = candidates.reset_index(drop=True).copy()
    candidates["_candidate_order"] = np.arange(len(candidates))
    candidates["_duplicate_group"] = _duplicate_components(candidates)
    candidates["_source_priority"] = candidates["_source_name"].map(_SOURCE_PRIORITY)
    candidates["_temperature_distance"] = np.where(
        candidates["_source_name"].eq("liverpool"),
        (
            pd.to_numeric(candidates["_temperature_c"], errors="coerce")
            - room_temperature_c
        ).abs(),
        0.0,
    )

    ranked = candidates.sort_values(
        [
            "_duplicate_group",
            "_source_priority",
            "_temperature_distance",
            "_source_row",
        ],
        kind="stable",
    )
    kept_indices = set(ranked.groupby("_duplicate_group", sort=False).head(1).index)
    kept = candidates.loc[sorted(kept_indices)].copy()
    dropped = candidates.loc[~candidates.index.isin(kept_indices)].copy()
    kept_by_group = kept.set_index("_duplicate_group")
    if not dropped.empty:
        dropped["removed_reason"] = dropped["_duplicate_group"].map(
            lambda group: (
                "duplicate DOI+formula; kept "
                f"{kept_by_group.at[group, 'ID']} from "
                f"{kept_by_group.at[group, '_source_name']}"
            )
        )

    audit_rows = []
    for group_id, group in candidates.groupby("_duplicate_group", sort=False):
        if len(group) <= 1:
            continue
        selected = kept_by_group.loc[group_id]
        values = pd.to_numeric(group["_conductivity_value"], errors="coerce")
        audit_rows.append({
            "audit_stage": "doi_formula",
            "duplicate_group": int(group_id),
            "Reduced Composition": selected["Reduced Composition"],
            "normalized_DOIs": "|".join(sorted({
                doi for tokens in group["_doi_tokens"] for doi in tokens
            })),
            "candidate_rows": int(len(group)),
            "sources": "|".join(group["_source_name"].astype(str)),
            "IDs": "|".join(group["ID"].astype(str)),
            "conductivities": "|".join(group["Ionic conductivity (S cm-1)"].astype(str)),
            "log10_min": float(np.log10(values.min())),
            "log10_max": float(np.log10(values.max())),
            "log10_span": float(np.log10(values.max()) - np.log10(values.min())),
            "kept_source": selected["_source_name"],
            "kept_ID": selected["ID"],
            "kept_conductivity": float(selected["_conductivity_value"]),
            "dropped_rows": int(len(group) - 1),
        })
    audit = pd.DataFrame(audit_rows)
    if not audit.empty:
        audit = audit.sort_values(
            ["log10_span", "candidate_rows"],
            ascending=[False, False],
            ignore_index=True,
        )
    return kept, dropped, audit


def _resolve_formula_span_conflicts(
    candidates: pd.DataFrame,
    span_limit: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """For large same-formula spans, retain only the highest conductivity row."""
    dropped_indices: set[int] = set()
    audit_rows = []

    for formula, group in candidates.groupby("Reduced Composition", sort=True):
        if len(group) <= 1:
            continue
        values = pd.to_numeric(group["_conductivity_value"], errors="coerce")
        log10_span = float(np.log10(values.max()) - np.log10(values.min()))
        if log10_span <= span_limit:
            continue

        ranked = group.assign(_rank_conductivity=values).sort_values(
            [
                "_rank_conductivity",
                "_source_priority",
                "_temperature_distance",
                "_source_row",
            ],
            ascending=[False, True, True, True],
            kind="stable",
        )
        selected = ranked.iloc[0]
        group_dropped_indices = set(group.index) - {selected.name}
        dropped_indices.update(group_dropped_indices)
        audit_rows.append({
            "audit_stage": "formula_span",
            "duplicate_group": "",
            "Reduced Composition": formula,
            "normalized_DOIs": "|".join(sorted({
                doi for tokens in group["_doi_tokens"] for doi in tokens
            })),
            "candidate_rows": int(len(group)),
            "sources": "|".join(group["_source_name"].astype(str)),
            "IDs": "|".join(group["ID"].astype(str)),
            "conductivities": "|".join(
                group["Ionic conductivity (S cm-1)"].astype(str)
            ),
            "log10_min": float(np.log10(values.min())),
            "log10_max": float(np.log10(values.max())),
            "log10_span": log10_span,
            "kept_source": selected["_source_name"],
            "kept_ID": selected["ID"],
            "kept_conductivity": float(selected["_conductivity_value"]),
            "dropped_rows": int(len(group_dropped_indices)),
        })

    dropped = candidates.loc[sorted(dropped_indices)].copy()
    kept = candidates.loc[~candidates.index.isin(dropped_indices)].copy()
    if not dropped.empty:
        selected_by_formula = {
            row["Reduced Composition"]: row
            for row in audit_rows
        }
        dropped["removed_reason"] = dropped["Reduced Composition"].map(
            lambda formula: (
                "conflicting formula: log10 conductivity span "
                f"{selected_by_formula[formula]['log10_span']:.3f} > "
                f"{span_limit:g}; kept highest conductivity row "
                f"{selected_by_formula[formula]['kept_ID']}"
            )
        )

    audit = pd.DataFrame(audit_rows)
    if not audit.empty:
        audit = audit.sort_values(
            ["log10_span", "candidate_rows"],
            ascending=[False, False],
            ignore_index=True,
        )
    return kept, dropped, audit


def prepare_absolute_v2_data(
    config: PrepareAbsoluteV2Config | None = None,
) -> PrepareAbsoluteResult:
    """Merge three databases and deduplicate by normalized DOI plus formula."""
    config = config or PrepareAbsoluteV2Config()
    source_frames = [
        _v1_candidates(config.v1_input_path),
        _caltech_candidates(config.caltech_input_path),
        _liverpool_candidates(config.liverpool_input_path),
    ]
    source_counts = {
        name: int(len(frame))
        for name, frame in zip(("v1", "caltech", "liverpool"), source_frames)
    }
    candidates = pd.concat(source_frames, ignore_index=True, sort=False)
    valid, invalid = _validate_v2_candidates(candidates, config)
    valid = _fill_caltech_families(valid)
    kept, duplicates, audit = _deduplicate_v2_candidates(
        valid,
        config.liverpool_room_temperature_c,
    )
    formula_span_duplicates = pd.DataFrame()
    formula_span_audit = pd.DataFrame()
    if config.keep_highest_for_conflicting_formulas:
        kept, formula_span_duplicates, formula_span_audit = (
            _resolve_formula_span_conflicts(
                kept,
                config.formula_log10_span_limit,
            )
        )
    kept, family_audit, family_counts = _label_unknown_families(kept, valid)
    audit = pd.concat([audit, formula_span_audit], ignore_index=True, sort=False)
    if not audit.empty:
        audit = audit.sort_values(
            ["log10_span", "candidate_rows"],
            ascending=[False, False],
            ignore_index=True,
        )
    output = kept.sort_values("_candidate_order")[ABSOLUTE_COLUMNS].reset_index(drop=True)
    excluded = pd.concat(
        [invalid, duplicates, formula_span_duplicates],
        ignore_index=True,
        sort=False,
    )
    excluded_output = excluded.reindex(
        columns=[*ABSOLUTE_COLUMNS, "removed_reason"]
    ).reset_index(drop=True)
    output_source_counts = kept["_source_name"].value_counts().to_dict()
    summary = {
        "config": asdict(config),
        "source_rows": source_counts,
        "candidate_rows": int(len(candidates)),
        "valid_candidate_rows": int(len(valid)),
        "output_rows": int(len(output)),
        "output_rows_by_source": {
            source: int(output_source_counts.get(source, 0))
            for source in ("v1", "caltech", "liverpool")
        },
        "excluded_rows": int(len(excluded_output)),
        "invalid_or_non_room_temperature_rows": int(len(invalid)),
        "duplicate_rows_removed": int(len(duplicates)),
        "duplicate_groups": int(
            audit["audit_stage"].eq("doi_formula").sum() if not audit.empty else 0
        ),
        "formula_span_groups": int(len(formula_span_audit)),
        "formula_span_rows_removed": int(len(formula_span_duplicates)),
        "family_rows_labeled": int(len(family_audit)),
        "remaining_unknown_family_rows": int(
            kept["Family"].map(normalize_family).eq("unknown").sum()
        ),
        "family_assignment_by_confidence": {
            str(confidence): int(count)
            for confidence, count in family_audit["confidence"].value_counts().items()
        },
        "deduplication_key": (
            "normalized DOI token overlap + normalized reduced formula; "
            "then reduced formula span"
        ),
        "selection_priority": [
            "DOI+formula: v1, caltech, liverpool closest to 25 C",
            "formula span: highest conductivity",
        ],
    }
    return PrepareAbsoluteResult(
        table=output,
        excluded=excluded_output,
        duplicate_audit=audit,
        summary=summary,
        family_audit=family_audit,
        family_counts=family_counts,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build absolute conductivity data.")
    parser.add_argument("--version", choices=("v1", "v2"), default="v1")
    args = parser.parse_args()
    if args.version == "v2":
        config = PrepareAbsoluteV2Config()
        result = prepare_absolute_v2_data(config)
    else:
        config = PrepareAbsoluteConfig()
        result = prepare_absolute_data(config)
    result.to_files(config)
    print(f"source_rows={result.summary['source_rows']}")
    print(f"output_rows={result.summary['output_rows']}")
    print(f"excluded_rows={result.summary['excluded_rows']}")
    if args.version == "v2":
        print(f"duplicate_groups={result.summary['duplicate_groups']}")
        print(f"output_rows_by_source={result.summary['output_rows_by_source']}")
    else:
        print(
            "conflicting_duplicate_formula_groups="
            f"{result.summary['conflicting_duplicate_formula_groups']}"
        )
    print(f"Output CSV   : {config.output_path.resolve()}")
    print(f"Excluded CSV : {config.excluded_path.resolve()}")
    print(f"Audit CSV    : {config.audit_path.resolve()}")


if __name__ == "__main__":
    main()
