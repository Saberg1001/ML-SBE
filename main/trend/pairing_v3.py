"""Deterministic trend pairing built directly on the absolute-v2 table.

Unlike ``pairing_simple.py``, which consumes a dedicated trend source table,
this module pairs rows from ``data/absolute/data-absolute-v2.csv`` (the raw
wide table, *not* the cleaned one-row-per-formula output).  Pairing is strictly
intra-literature: rows are grouped by DOI + family and only connected within a
group, so no conductivity difference is ever drawn across measurement campaigns.

Cleaning policy (mirrors the absolute model threshold):
* keep only room-temperature measurements (25-30 C, or the tokens RT/25/27/30);
* drop explicit extrapolated-temperature records;
* drop conductivities below the absolute-model threshold (1e-6 S/cm).

Within a group:
* identical element sets -> composition/concentration series, adjacent pairs;
* changing element sets -> element substitution, all C(n,2) ordered pairs.

The emitted target is the log-space change:
    delta_log10_IC = log10(IC_B) - log10(IC_A)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

if __package__ is None:
    _PROJECT_ROOT = Path(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))
else:
    _PROJECT_ROOT = Path(__file__).resolve().parents[2]

import itertools
import re

import numpy as np
import pandas as pd
from pymatgen.core import Composition

from ..paths import DATA_DIR
from ..features import normalize_family


ABSOLUTE_V2_CSV = DATA_DIR / "absolute" / "data-absolute-v2.csv"
DEFAULT_OUTPUT = DATA_DIR / "trend" / "data-trend-v3-pairs.csv"

TREND_MIN_CONDUCTIVITY_S_CM = 1e-6
# Room-temperature window for records that carry a numeric Celsius value.
ROOM_TEMP_MIN_C = 25.0
ROOM_TEMP_MAX_C = 30.0
# Non-numeric tokens (from ``measured_temperature=``) accepted as room temp.
ROOM_TEMP_TOKENS = {"rt", "25", "27", "30"}
# Substrings that mark an explicitly extrapolated (non-RT) measurement.
EXTRAPOLATION_MARKERS = ("外推", "extrapolat", "extrap")


def _formula_parts(formula: str) -> tuple[tuple[str, ...], tuple[float, ...]]:
    comp = Composition(formula)
    amounts = comp.get_el_amt_dict()
    elements = tuple(sorted(amounts))
    total = sum(amounts.values())
    vector = tuple(amounts[e] / total for e in elements)
    return elements, vector


def _parse_conductivity(value: object) -> float:
    text = re.sub(r"^[≈~]", "", str(value).strip()).strip()
    return float(text)


def _celsius_from_note(note: str) -> float | None:
    """Return the numeric Celsius value from ``measured_temperature_C=``, or None."""
    m = re.search(r"measured_temperature_C=([^;]+)", note)
    if m:
        try:
            v = float(m.group(1))
            return v if 0 < v <= 100 else None
        except ValueError:
            return None
    return None


def _token_from_note(note: str) -> str:
    """Return the raw token from ``measured_temperature=``, lower-cased."""
    m = re.search(r"measured_temperature=([^;]+)", note)
    return re.sub(r"[°\s]", "", m.group(1)).lower() if m else ""


def _is_room_temperature(note: str) -> bool:
    """True for room-temperature rows; rows with no temperature info are kept."""
    lower = note.lower()
    if any(marker in lower for marker in EXTRAPOLATION_MARKERS):
        return False

    celsius = _celsius_from_note(note)
    if celsius is not None:
        return ROOM_TEMP_MIN_C <= celsius <= ROOM_TEMP_MAX_C

    token = _token_from_note(note)
    if token:
        # Accept "25", "27", "30", "rt", and strip trailing degree or unit chars.
        return token in ROOM_TEMP_TOKENS

    # No temperature annotation at all: old literature that only states RT.
    return True


def clean_source(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter to room-temperature, above-threshold rows.

    Returns (kept, dropped) with a ``removed_reason`` column on dropped.
    """
    conductivity = df["Ionic conductivity (S cm-1)"].map(_parse_conductivity)
    low_mask = conductivity < TREND_MIN_CONDUCTIVITY_S_CM
    temp_mask = ~df["note"].map(_is_room_temperature)

    remove_reason = pd.Series("kept", index=df.index)
    remove_reason[low_mask] = "low_conductivity"
    # Temperature filter takes precedence in labelling for the dropped frame.
    remove_reason[temp_mask] = "non_room_temperature"

    kept = df[remove_reason == "kept"].copy()
    dropped = df[remove_reason != "kept"].copy()
    dropped["removed_reason"] = remove_reason[remove_reason != "kept"]
    return kept, dropped


def build_pairs(source: pd.DataFrame) -> pd.DataFrame:
    """Return the v3 pair table from a cleaned absolute-v2 frame."""
    required = {"ID", "Reduced Composition", "Ionic conductivity (S cm-1)",
                "Family", "DOI", "note"}
    missing = sorted(required - set(source.columns))
    if missing:
        raise ValueError(f"Source table is missing columns: {missing}")

    frame = source.copy().reset_index(drop=True)

    info = frame["Reduced Composition"].map(_formula_parts)
    frame["_element_set"] = info.map(lambda x: x[0])
    frame["_composition_vector"] = info.map(lambda x: x[1])
    frame["_family"] = frame["Family"].map(normalize_family)
    frame["_ic"] = frame["Ionic conductivity (S cm-1)"].map(_parse_conductivity)
    frame["_log10_ic"] = np.log10(frame["_ic"])

    group_cols = ["_family", "DOI"]
    frame = frame.sort_values(group_cols + ["ID"], kind="stable")

    records: list[dict] = []
    group_number = 0
    pair_number = 0

    for _, group in frame.groupby(group_cols, sort=True, dropna=False):
        group = group.sort_values(["_composition_vector", "ID"], kind="stable")
        group_number += 1
        group_id = f"trgrp_v3_{group_number:04d}"
        rows = list(group.iterrows())

        element_sets = {row["_element_set"] for _, row in rows}
        if len(element_sets) == 1:
            pair_iter = zip(rows, rows[1:])
            strategy = "adjacent_concentration"
        else:
            pair_iter = itertools.combinations(rows, 2)
            strategy = "all_element_substitutions"

        for (_, row_a), (_, row_b) in pair_iter:
            if row_a["Reduced Composition"] == row_b["Reduced Composition"]:
                continue
            pair_number += 1
            records.append({
                "group_id": group_id,
                "pair_id": f"trpair_v3_{pair_number:06d}",
                "pairing_strategy": strategy,
                "formula_a": row_a["Reduced Composition"],
                "formula_b": row_b["Reduced Composition"],
                "conductivity_a_S_cm-1": row_a["_ic"],
                "conductivity_b_S_cm-1": row_b["_ic"],
                "log10_IC_a": row_a["_log10_ic"],
                "log10_IC_b": row_b["_log10_ic"],
                "delta_log10_IC": float(row_b["_log10_ic"] - row_a["_log10_ic"]),
                "family": row_a["_family"],
                "doi": row_a["DOI"],
                "id_a": row_a["ID"],
                "id_b": row_b["ID"],
            })

    return pd.DataFrame(records, columns=[
        "group_id", "pair_id", "pairing_strategy",
        "formula_a", "formula_b",
        "conductivity_a_S_cm-1", "conductivity_b_S_cm-1",
        "log10_IC_a", "log10_IC_b", "delta_log10_IC",
        "family", "doi", "id_a", "id_b",
    ])


def main() -> None:
    """Pair absolute-v2 into trend-v3 pairs (point-run entry).

    Run directly via ``python main/trend/pairing_v3.py`` (the "Run" button).
    Input  : data/absolute/data-absolute-v2.csv
    Output : data/trend/data-trend-v3-pairs.csv
    """
    source = pd.read_csv(ABSOLUTE_V2_CSV, dtype=str, keep_default_na=False)
    kept, dropped = clean_source(source)
    pairs = build_pairs(kept)
    DEFAULT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(DEFAULT_OUTPUT, index=False)
    print(f"source_rows={len(source)}  kept={len(kept)}  dropped={len(dropped)}")
    print(f"pair_rows={len(pairs)}  groups={pairs['group_id'].nunique() if len(pairs) else 0}")
    print(f"delta_log10_IC  mean={pairs['delta_log10_IC'].mean():.2f}  "
          f"std={pairs['delta_log10_IC'].std():.2f}  "
          f"min={pairs['delta_log10_IC'].min():.2f}  "
          f"max={pairs['delta_log10_IC'].max():.2f}")
    print(f"Output CSV : {DEFAULT_OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
