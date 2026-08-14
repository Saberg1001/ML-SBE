"""Simple, deterministic trend pairing for an already-clean source table.

Pairing policy:
* identical element sets: treat as a composition/concentration series and
  connect adjacent compositions;
* changing element sets: treat as element substitution and emit all pairs.

The script deliberately does not clean or alter source rows.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Allow running this module directly as a script (VS Code "Run" button): expose
# the project root so path defaults resolve regardless of the working directory.
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

import pandas as pd
from pymatgen.core import Composition


DEFAULT_INPUT = _PROJECT_ROOT / "data/trend/data-trend-v1-clean.csv"
DEFAULT_OUTPUT = _PROJECT_ROOT / "data/trend/data-trend-v1-pairs.csv"


def _formula_info(formula: str) -> tuple[tuple[str, ...], tuple[float, ...]]:
    comp = Composition(formula)
    amounts = comp.get_el_amt_dict()
    elements = tuple(sorted(amounts))
    # Normalized composition makes adjacent ordering independent of formula
    # formatting and total formula-unit size.
    total = sum(amounts.values())
    vector = tuple(amounts[e] / total for e in elements)
    return elements, vector


def _measurement_type(note: str) -> str:
    text = str(note).casefold()
    has_bulk = "σbulk" in text or "sb​​ulk" in text or "bulk" in text
    has_total = "σtot" in text or "total" in text
    if has_bulk and not has_total:
        return "bulk"
    if has_total and not has_bulk:
        return "tot"
    return "unknown"


def _conductivity_value(value: str) -> float:
    """Parse a cleaned single-value conductivity in S/cm."""
    text = re.sub(r"^[≈~]", "", str(value).strip()).strip()
    return float(text)


def _sort_key(row: pd.Series) -> tuple:
    return tuple(row["composition_vector"])


def build_pairs(source: pd.DataFrame) -> pd.DataFrame:
    required = {
        "ID", "化学式", "电导率", "测得温度", "化合物制备方法",
        "family", "对应文献DOI",
    }
    missing = sorted(required - set(source.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    frame = source.copy().reset_index(drop=True)
    info = frame["化学式"].map(_formula_info)
    frame["element_set"] = info.map(lambda x: x[0])
    frame["composition_vector"] = info.map(lambda x: x[1])
    frame["measurement_type"] = frame.get(
        "备注", pd.Series("", index=frame.index)
    ).map(_measurement_type)
    group_columns = [
        "对应文献DOI", "化合物制备方法", "测得温度", "family",
        "measurement_type",
    ]
    frame = frame.sort_values(group_columns + ["ID"], kind="stable")
    records: list[dict] = []
    group_number = 0
    pair_number = 0
    for _, group in frame.groupby(group_columns, sort=True, dropna=False):
        group = group.sort_values(["composition_vector", "ID"], kind="stable")
        group_number += 1
        group_id = f"trgrp_{group_number:04d}"
        rows = list(group.iterrows())
        element_sets = {row["element_set"] for _, row in rows}
        if len(element_sets) == 1:
            pairs = zip(rows, rows[1:])
            strategy = "adjacent_concentration"
        else:
            pairs = itertools.combinations(rows, 2)
            strategy = "all_element_substitutions"
        for (idx_a, row_a), (idx_b, row_b) in pairs:
            if row_a["化学式"] == row_b["化学式"]:
                continue
            pair_number += 1
            records.append({
                "group_id": group_id,
                "pair_id": f"trpair_{pair_number:06d}",
                "化学式_a": row_a["化学式"],
                "化学式_b": row_b["化学式"],
                "电导率_a": row_a["电导率"],
                "电导率_b": row_b["电导率"],
                "电导率变化值": _conductivity_value(row_b["电导率"])
                - _conductivity_value(row_a["电导率"]),
                "温度": row_a["测得温度"],
                "合成方法": row_a["化合物制备方法"],
                "family": row_a["family"],
                "doi": row_a["对应文献DOI"],
            })
    return pd.DataFrame(records, columns=[
        "group_id", "pair_id", "化学式_a", "化学式_b", "电导率_a",
        "电导率_b", "电导率变化值", "温度", "合成方法", "family", "doi",
    ])


def main() -> None:
    """Pair the cleaned trend source table (point-run entry).

    Run directly via ``python main/trend/pairing_simple.py`` (the "Run" button).
    Input  : data/trend/data-trend-v1-clean.csv
    Output : data/trend/data-trend-v1-pairs.csv
    """
    source = pd.read_csv(DEFAULT_INPUT, dtype=str, keep_default_na=False)
    pairs = build_pairs(source)
    DEFAULT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(DEFAULT_OUTPUT, index=False)
    print(f"source_rows={len(source)} pair_rows={len(pairs)} "
          f"groups={pairs['group_id'].nunique() if len(pairs) else 0}")
    print(f"Output CSV : {DEFAULT_OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
