"""Build the single lab-experiment summary table from the three raw sources.

Sources
-------
* data/experimental/raw/experimental-data.csv  -> exp_001..exp_113 (sulfides)
* data/experimental/raw/without-P.csv          -> exp_114..exp_125 (P-free sulfides)
* data/experimental/raw/halide.csv             -> exp_126..exp_138 (halides)

Rules
-----
1. One row per measurement. Every row gets an `exp_NNN` ID, so a single ID
   space marks the whole file as in-house experimental data; the `来源文件`
   column keeps each row traceable to the raw file it came from. Sulfide rows
   keep the ID already written in experimental-data.csv; the P-free and halide
   rows continue the sequence in file order.
2. Rows whose composition carries a "+5wt%ZrCl4" additive are dropped: the
   additive is not part of the composition, so they cannot be compared or
   modelled alongside the bare hosts.
3. Duplicate formulas inside one source keep every measurement and get the same
   `dup_group` value, so the clash is visible instead of silently averaged.
4. `family` comes from the raw file when it has one, else from the literature
   registry (data-absolute-v3*.csv) matched on the pymatgen reduced formula.

Usage
-----
    python main/experimental/build_lab_summary.py
"""

from __future__ import annotations

import os
import re
import sys

if __package__ is None:
    _PROJECT_ROOT = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

from pathlib import Path

import pandas as pd

from main.experimental.compare_lab_vs_literature import (
    _canonical_formula,
    _parse_sci,
    _read_halide_lab,
    _read_sulfide_lab,
)
from main.paths import DATA_DIR

EXPERIMENTAL_DIR = DATA_DIR / "experimental"
RAW_DIR = EXPERIMENTAL_DIR / "raw"
OUTPUT_PATH = EXPERIMENTAL_DIR / "experimental-summary.csv"

COLUMNS = ["ID", "化学式", "电导率（S/cm）", "制备方式", "family", "来源文件", "备注", "dup_group"]


def _round_sig(value: float, digits: int = 4) -> float:
    """Round to `digits` significant figures (avoids 0.00037000000000000005)."""
    if value == 0 or pd.isna(value):
        return float(value)
    import math

    return round(value, -int(math.floor(math.log10(abs(value)))) + (digits - 1))
# The literature tables spell a few family labels inconsistently; keep one form.
FAMILY_ALIASES = {
    "argyrodite": "argyrodites",
    "argyrodites": "argyrodites",
    "halide": "halides",
    "halides": "halides",
    "lithium_argyrodite": "argyrodites",
    "thio_lisicon": "thio_lisicon",
    "l_g_p_s": "lgps",
    "lgps": "lgps",
}


def _literature_family_map() -> dict[str, str]:
    """Reduced formula -> Family, from every row of the literature tables."""
    frames = []
    for name in ("data-absolute-v3.csv", "data-absolute-v3-model-clean.csv"):
        path = DATA_DIR / "absolute" / name
        if path.exists():
            frame = pd.read_csv(path, dtype=str, keep_default_na=False)
            frames.append(frame[["True Composition", "Family"]])
    if not frames:
        return {}
    lit = pd.concat(frames, ignore_index=True)
    lit["canon"] = lit["True Composition"].map(_canonical_formula)
    lit = lit.dropna(subset=["canon"])
    lit = lit[lit["Family"].astype(str).str.strip() != ""]
    return lit.drop_duplicates("canon").set_index("canon")["Family"].to_dict()


def _normalize_family(label: str) -> str:
    """Collapse the literature table's spelling variants to one label."""
    text = str(label).strip()
    if not text:
        return ""
    key = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return FAMILY_ALIASES.get(key, key)


def _family_from_composition(formula: str) -> str:
    """Fallback family label when the literature tables have no matching row.

    Ordered rules, most specific first; the split follows the argyrodite
    (Li6PS5X-type), LGPS (Li10MP2S12-type) and thio-LISICON conventions.
    """
    text = str(formula)
    has_halogen = bool(re.search(r"Cl|Br|I", text))
    # Li10MP2S12 / Li9.54Si1.74P1.44S11.x (LGPS-type) must be tested before the
    # generic "P + sulfur" rule, since those formulas also match it.
    if re.search(r"^Li(9\.\d+|10)", text) and "P" in text:
        return "lgps"
    # Chloride/oxyhalide electrolytes: a halogen with no sulfide sulfur.
    if has_halogen and not re.search(r"S\d", text):
        return "halides"
    if "P" in text and re.search(r"S\d", text):
        return "argyrodites"
    # P-free Sb/Si/As sulfides: the "SbSi" argyrodite-type family.
    if re.search(r"Sb|As|Si|Sn", text) and re.search(r"S\d", text):
        return "argyrodites" if has_halogen else "thio_lisicon"
    return ""


def _method_from_note(note: str) -> str:
    """Normalize the raw method note into one of the summary's labels."""
    text = str(note).strip()
    if not text or text == "unknow":
        return "unknow"
    if "固相" in text:
        return "固相"
    if "球磨" in text:
        return "球磨"
    return text


def _build_rows() -> list[dict]:
    family_map = _literature_family_map()
    rows: list[dict] = []

    # --- sulfides (experimental-data.csv) and P-free series (without-P.csv) ---
    # experimental-data.csv rows keep their existing exp_NNN IDs.
    # without-P.csv rows are renumbered as exp_114, exp_115, ... in file order.
    nop_index = 0  # counts P-free rows for sequential ID assignment
    sulfide = _read_sulfide_lab()
    for _, row in sulfide.iterrows():
        if pd.isna(row["conductivity_S_cm"]):
            continue
        note = str(row.get("note", "")).strip()
        method = _method_from_note(note)
        family = _normalize_family(family_map.get(row["canonical"], ""))

        if row["table"] == "without-P.csv":
            # without-P.csv records the measurement condition in the value cell
            # ("14.123（打粉，25℃）"); read method from the note text instead.
            identifier = f"exp_{114 + nop_index:03d}"
            nop_index += 1
            method = "打粉" if "打粉" in note else ("unknow" if note in ("", "unknow") else note)
            # P-free Sb/Si argyrodites; literature table lacks most of them.
            family = family or "argyrodites"
        else:
            # experimental-data.csv: keep the existing exp_NNN ID.
            identifier = row["ID"]

        rows.append(
            {
                "ID": identifier,
                "化学式": row["formula"],
                "电导率（S/cm）": _round_sig(float(row["conductivity_S_cm"])),
                "制备方式": method,
                "family": family,
                "来源文件": row["table"],
                "备注": note,
            }
        )

    # --- halides (halide.csv) ---
    # Rows with "+" in the formula are additive composites ("+5wt%ZrCl4") and
    # are excluded (rule 2).  Clean rows are numbered exp_126, exp_127, ... in
    # file order (group1 tab-delimited first, then comma-delimited hal_NNN rows).
    hal_index = 0  # counts only non-additive rows
    halide = _read_halide_lab().reset_index(drop=True)
    for _, row in halide.iterrows():
        if pd.isna(row["conductivity_S_cm"]):
            continue
        if row["additive"]:
            continue  # rule 2: "+5wt%ZrCl4" type rows stay out of the summary
        identifier = f"exp_{126 + hal_index:03d}"
        hal_index += 1
        rows.append(
            {
                "ID": identifier,
                "化学式": row["formula"],
                "电导率（S/cm）": _round_sig(float(row["conductivity_S_cm"]), 4),
                "制备方式": _method_from_note(row.get("note", "")),
                "family": _normalize_family(family_map.get(row["canonical"], "halides")),
                "来源文件": row["table"],
                "备注": str(row.get("note", "")).strip(),
            }
        )
    return rows


def main() -> None:
    frame = pd.DataFrame(_build_rows())

    frame["canonical"] = frame["化学式"].map(_canonical_formula)
    duplicated = frame["canonical"].duplicated(keep=False)
    frame["dup_group"] = ""
    for canon in frame.loc[duplicated, "canonical"].dropna().unique():
        members = frame.index[frame["canonical"] == canon]
        frame.loc[members, "dup_group"] = f"dup_{canon}"
    frame = frame.drop(columns=["canonical"])

    frame = frame[COLUMNS]
    frame.to_csv(OUTPUT_PATH, index=False)

    print(f"output   : {OUTPUT_PATH}")
    print(f"rows     : {len(frame)}")
    print(frame.groupby("来源文件").size().to_string())
    print()
    print("family:")
    print(frame["family"].replace("", "(unknown)").value_counts().to_string())
    print()
    dupes = frame[frame["dup_group"] != ""][["dup_group", "ID", "化学式", "电导率（S/cm）", "来源文件"]]
    print(f"duplicate formulas ({dupes['dup_group'].nunique() if len(dupes) else 0} groups, {len(dupes)} rows):")
    print(dupes.to_string(index=False) if len(dupes) else "(none)")


if __name__ == "__main__":
    main()
