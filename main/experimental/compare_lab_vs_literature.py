"""Reproduce the halide lab-vs-literature comparison and extend it to sulfides.

Goal
----
The halide table in reports/experimental/experimental-halide-summary.csv compares
each in-house measured halide against the literature value for the SAME formula
(Delta log10 = measured - literature, in log10 S/cm). This script:
  1. reproduces that halide table from the raw experimental files,
  2. does the same for the sulfide lab tables (experimental-data.csv, without-P.csv)
     against data-absolute-v3-model-clean.csv / data-absolute-v3.csv,
  3. reports the offset statistics and flags the pairs that need a data-cleaning
     decision (glass-ceramic vs crystalline phase, method offset, unknown phase).

Usage:
    python main/experimental/compare_lab_vs_literature.py [--out reports/experimental]
"""

from __future__ import annotations

import argparse
import json
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

import numpy as np
import pandas as pd
from pymatgen.core import Composition

from main.paths import DATA_DIR

EXPERIMENTAL_DIR = DATA_DIR / "experimental"
HALIDE_RAW = EXPERIMENTAL_DIR / "raw" / "halide.csv"
SULFIDE_RAW = EXPERIMENTAL_DIR / "raw" / "experimental-data.csv"
NOP_RAW = EXPERIMENTAL_DIR / "raw" / "without-P.csv"
V3_CLEAN = DATA_DIR / "absolute" / "data-absolute-v3-model-clean.csv"
V3_ALL = DATA_DIR / "absolute" / "data-absolute-v3.csv"

# Matrix (glassy / glass-ceramic) hosts: their literature value describes a
# different phase than the crystallized in-house sample, so a large negative
# Delta is expected and is NOT evidence of a data error.
GLASS_MARKERS = ("glass", "glass_ceramic")
UNKNOWN_COMPOSITION = {
    "Li6.6Sb0.4Si0.6S5I": "exp_033 formula written Li6.6Sb0.4Si0.6S5I but the surrounding "
    "series is Li6.7Sb0.3Si0.7S5I (x=0.4 -> Li6.6Sb0.4Si0.6); verify Li count",
}


def _log10(value: float) -> float:
    return float(np.log10(value)) if value and value > 0 else float("nan")


def _canonical_formula(text: str) -> str | None:
    """Normalize a formula to a pymatgen reduced formula for matching."""
    cleaned = str(text).split("#")[0].strip()
    cleaned = cleaned.replace("−", "-").replace(" ", "")
    if not cleaned:
        return None
    try:
        return Composition(cleaned).reduced_formula
    except Exception:
        return None


_SCI_RE = re.compile(r"([\d.]+)\s*[x×]\s*10\s*\^?\s*-\s*(\d+)")


def _parse_sci(text: str) -> float:
    """Parse "1.5 x 10-3" or plain "0.00083" conductivity values."""
    match = _SCI_RE.search(text)
    if match:
        return float(match.group(1)) * 10 ** (-int(match.group(2)))
    try:
        return float(text)
    except ValueError:
        return float("nan")


def _read_halide_lab() -> pd.DataFrame:
    """Parse data/experimental/raw/halide.csv (tab groups + comma hal_ rows).

    The formula keeps its additive suffix ("+ 5wt%ZrCl4") in `formula` for
    display, while `canonical` is taken from the bare host formula so it can be
    matched against the literature tables.
    """
    rows = []
    current_group = ""
    for line in HALIDE_RAW.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or "电导率" in line or line.startswith("组成"):
            continue
        if line.lower().startswith("group") and "\t" not in line:
            current_group = line
            continue
        # Layout 1: "<hal_ID>,<formula>,<value>,<note>,<family>" (comma).
        # Layout 2: "<formula>\t<value>\t\t<note>" (tab, inside a group).
        if "\t" in line:
            cells = [cell.strip() for cell in line.split("\t") if cell.strip()]
            if len(cells) < 2:
                continue
            identifier = f"{current_group or 'hal'}_{len(rows) + 1:03d}"
            formula = cells[0].split("#")[0].strip()
            value = cells[1].split("#")[0].strip()
            note = cells[2] if len(cells) > 2 else ""
        else:
            cells = [cell.strip() for cell in line.split(",")]
            if len(cells) < 3:
                continue
            identifier, formula, value = cells[0], cells[1], cells[2]
            note = cells[3] if len(cells) > 3 else ""
        rows.append(
            {
                "table": "halide.csv",
                "group": current_group,
                "ID": identifier,
                "formula": formula,
                "note": note,
                "conductivity_S_cm": _parse_sci(value),
            }
        )
    frame = pd.DataFrame(rows)
    frame["additive"] = frame["formula"].str.contains(r"\+", regex=True)
    frame["canonical"] = frame["formula"].map(
        lambda text: _canonical_formula(text.split("+")[0])
    )
    return frame


def _read_sulfide_lab() -> pd.DataFrame:
    """Parse both sulfide lab tables into (source, ID, formula, conductivity)."""
    rows = []
    current_group = ""
    for line in SULFIDE_RAW.read_text(encoding="utf-8").splitlines():
        parts = [part.strip() for part in line.split("\t") if part.strip()]
        if len(parts) == 1:
            current_group = parts[0]
            continue
        if len(parts) < 3:
            continue
        identifier, formula, conductivity = parts[0], parts[1], parts[2]
        rows.append(
            {
                "table": "experimental-data.csv",
                "group": current_group,
                "ID": identifier,
                "formula": formula.split("#")[0].strip(),
                "note": parts[2].split("#", 1)[1].strip() if "#" in parts[2] else "",
                "conductivity_S_cm": pd.to_numeric(conductivity.split("#")[0].strip(), errors="coerce"),
            }
        )

    for line in NOP_RAW.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|") or line.startswith("|---") or "样品分子式" in line:
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) < 3:
            continue
        formula = cells[1].translate(str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789"))
        value = pd.to_numeric(re.sub(r"[^\d.]", "", cells[2].split("（")[0]), errors="coerce")
        rows.append(
            {
                "table": "without-P.csv",
                "group": cells[0],
                "ID": "nop_%02d" % (len(rows) + 1),
                "formula": formula,
                "note": cells[2] if "（" in cells[2] else "",
                "conductivity_S_cm": (value / 1000.0) if pd.notna(value) else np.nan,
            }
        )
    frame = pd.DataFrame(rows)
    frame["additive"] = frame["formula"].str.contains(r"\+")
    frame["canonical"] = frame["formula"].map(_canonical_formula)
    return frame


def _literature_table() -> pd.DataFrame:
    """Clean v3 table + every v3 row that lost the one-row-per-formula vote."""
    clean = pd.read_csv(V3_CLEAN, dtype=str, keep_default_na=False)
    allv3 = pd.read_csv(V3_ALL, dtype=str, keep_default_na=False)
    lit = clean[["Reduced Composition", "True Composition", "Family", "DOI", "Ref",
                 "Ionic conductivity (S cm-1)", "note"]].copy()
    lit["in_clean_table"] = True
    for column in ("Reduced Composition", "Ionic conductivity (S cm-1)"):
        assert column in allv3.columns
    extra = allv3[~allv3["True Composition"].isin(clean["True Composition"])].copy()
    extra["Reduced Composition"] = extra["Reduced Composition"].map(_canonical_formula)
    extra = extra[["Reduced Composition", "True Composition", "Family", "DOI", "Ref",
                   "Ionic conductivity (S cm-1)", "note"]]
    extra["in_clean_table"] = False
    lit = pd.concat([lit, extra], ignore_index=True)
    lit["canonical"] = lit["Reduced Composition"].map(_canonical_formula)
    lit["lit_S_cm"] = pd.to_numeric(lit["Ionic conductivity (S cm-1)"], errors="coerce")
    lit["lit_log10"] = lit["lit_S_cm"].map(_log10)
    lit["lit_phase"] = lit["note"].str.extract(r"synthesis_method=([^;]+)")[0].fillna("")
    lit["lit_is_glass"] = lit["Family"].isin(GLASS_MARKERS)
    return lit


def _compare(lab: pd.DataFrame, lit: pd.DataFrame, lab_label: str) -> pd.DataFrame:
    best = (
        lit.dropna(subset=["canonical", "lit_log10"])
        .sort_values("lit_log10", ascending=False)
        .drop_duplicates("canonical", keep="first")
        .set_index("canonical")
    )
    grouped = lit.dropna(subset=["canonical", "lit_log10"]).groupby("canonical").agg(
        n_lit=("lit_log10", "size"),
        lit_log10_min=("lit_log10", "min"),
        lit_log10_max=("lit_log10", "max"),
        lit_span=("lit_log10", lambda values: float(values.max() - values.min())),
    )
    rows = []
    for _, row in lab.iterrows():
        key = row["canonical"]
        record = {
            "table": row["table"],
            "group": row["group"],
            "ID": row["ID"],
            "formula": row["formula"],
            "note": row.get("note", ""),
            "measured_log10": _log10(row["conductivity_S_cm"]),
            "flag": "",
        }
        if key is None or key not in best.index:
            record.update({"matched_clean_formula": "", "DOI": "", "lit_log10": np.nan,
                           "delta_log10": np.nan, "n_lit": 0,
                           "flag": "no literature entry for this formula"})
            rows.append(record)
            continue
        reference = best.loc[key]
        record.update(
            {
                "matched_clean_formula": key,
                "DOI": reference["DOI"],
                "lit_log10": float(reference["lit_log10"]),
                "delta_log10": float(record["measured_log10"] - reference["lit_log10"]),
                "n_lit": int(grouped.loc[key, "n_lit"]),
                "lit_spread": float(grouped.loc[key, "lit_span"]),
                "lit_family": reference["Family"],
                "lit_phase": reference["lit_phase"],
            }
        )
        if record["lit_family"] in GLASS_MARKERS:
            record["flag"] = "literature value is a glass/glass-ceramic; phase mismatch"
        elif abs(record["delta_log10"]) >= 0.7:
            record["flag"] = "|delta| >= 0.7 (~5x): check composition or method"
        if row.get("additive"):
            record["flag"] = (record["flag"] + "; " if record["flag"] else "") + (
                "lab sample has an additive (e.g. 5wt%ZrCl4); literature row is the bare host"
            )
        record["lab_kind"] = lab_label
        rows.append(record)
    out = pd.DataFrame(rows)
    for column in ("matched_clean_formula", "DOI", "lit_family", "lit_phase", "flag", "lab_kind"):
        if column in out.columns:
            out[column] = out[column].fillna("")
    for column in ("n_lit", "lit_spread"):
        if column in out.columns:
            out[column] = out[column].fillna(0)
    return out


def _summary(frame: pd.DataFrame) -> dict:
    paired = frame.dropna(subset=["delta_log10"])
    delta = paired["delta_log10"]
    summary = {
        "pairs": int(len(paired)),
        "unmatched": int(frame["delta_log10"].isna().sum()),
        "median_delta_log10": float(delta.median()),
        "mean_delta_log10": float(delta.mean()),
        "std_delta_log10": float(delta.std(ddof=1)) if len(delta) > 1 else float("nan"),
        "p05_delta": float(delta.quantile(0.05)),
        "p95_delta": float(delta.quantile(0.95)),
        "share_lab_lower_than_lit": float((delta < 0).mean()),
        "share_within_0.3": float((delta.abs() <= 0.3).mean()),
        "share_beyond_0.7": float((delta.abs() >= 0.7).mean()),
        "median_measured_log10": float(paired["measured_log10"].median()),
        "median_lit_log10": float(paired["lit_log10"].median()),
        "median_lit_spread_per_formula": float(
            paired["lit_spread"].median() if "lit_spread" in paired else float("nan")
        ),
    }
    # Method-stratified offset: ball milling vs solid state is the one factor
    # that separates the halide pairs cleanly, so report it whenever known.
    if "note" in paired.columns:
        known = paired[paired["note"].astype(str).str.strip() != ""]
        groups = {}
        for label, part in known.groupby(known["note"].astype(str).str.strip()):
            if len(part) >= 2:
                groups[label] = {
                    "n": int(len(part)),
                    "median_delta_log10": float(part["delta_log10"].median()),
                }
        if groups:
            summary["by_method"] = groups
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="reports/experimental")
    args = parser.parse_args()

    lit = _literature_table()
    lab_frames = {
        "sulfide": _read_sulfide_lab(),
        "halide": _read_halide_lab(),
    }
    frames = {name: _compare(frame, lit, name) for name, frame in lab_frames.items()}

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, frame in frames.items():
        frame.drop(columns=["lab_kind"]).to_csv(out_dir / f"{name}-vs-literature.csv", index=False)
        summary[name] = _summary(frame)
    (out_dir / "lab-vs-literature-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    for name, frame in frames.items():
        print(f"=== {name} ===")
        print(frame.drop(columns=["lab_kind"]).to_string(index=False))
    print("\n", json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
