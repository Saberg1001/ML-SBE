"""Produce data-absolute-v3-model-clean.csv from data-absolute-v3.csv.

Selection rules
---------------
1. Remove rows with conductivity <= 0 or missing.
2. Remove rows below 1e-6 S/cm (model threshold), EXCEPT experimental rows
   (exp_NNN) which bypass this floor.
3. Liverpool and sulfide rows measured outside 20-30 C are removed; Liverpool
   rows with unknown temperature are also removed.
4. Per unique Reduced Composition, keep one row:
     a. If any experimental (exp_NNN) row exists → keep the exp row with the
        highest conductivity (experimental data = ground truth).
     b. Otherwise keep the literature row with the highest conductivity, with
        tie-breaking: checked > unchecked, then curated_lit > db_liverpool >
        db_caltech > db_obelix, then original row order.
   When multiple rows have the *exact same conductivity value*, they are
   considered replications of the same measurement: keep one, merge their
   DOIs into `all_dois`.
5. Two new columns added to the output:
     - `source_bucket`: coarse provenance label for sample weighting
         experimental   – rows from data/experimental/experimental-summary.csv
         curated_lit    – hand-curated tables: sulfide-clean, halide adddata,
                          literature_additions, Keggin
         db_liverpool   – Liverpool crystal-structure database
         db_caltech     – Caltech ICSD/AIMD-predicted table
         db_obelix      – OBeLiX v1 (the main v2 source)
     - `all_dois`: semicolon-separated list of all DOIs that reported this
         composition at this (or any merged) conductivity value; starts with
         the kept row's own DOI.

Usage
-----
    python main/absolute/clean_v3_weighted.py
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import sys
from pathlib import Path

if __package__ is None:
    _ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)

from main.features import normalize_family, parse_conductivity
from main.paths import DATA_DIR

ABSOLUTE_DIR = DATA_DIR / "absolute"
IN_PATH   = ABSOLUTE_DIR / "data-absolute-v3.csv"
OUT_PATH  = ABSOLUTE_DIR / "data-absolute-v3-model-clean.csv"
OUT_EXCL  = ABSOLUTE_DIR / "data-absolute-v3-model-clean-excluded.csv"
OUT_SUM   = ABSOLUTE_DIR / "data-absolute-v3-model-clean-summary.json"

MIN_CONDUCTIVITY = 1e-6
TEMP_MIN, TEMP_MAX = 20.0, 30.0

# Source buckets (order matters for tie-breaking: lower index = preferred)
BUCKET_RANK = {
    "experimental": 0,
    "curated_lit":  1,
    "db_obelix":    2,
    "db_liverpool": 3,
    "db_caltech":   4,
}

FORMULA_PREFERRED_FAMILY = {
    "Li1.6Al0.6Ge1.4P3O12": "nasicon",
    "Li10Sn(PS6)2": "lgps",
    "Li7P3S11": "thio_lisicon",
}
FORMULA_FAMILY_RELABEL = {
    "LiNbCl4O": "oxyhalides",
    "LiTaCl4O": "oxyhalides",
}


def _source_bucket(row_id: str, ref: str) -> str:
    if row_id.startswith("exp_"):
        return "experimental"
    ref_l = ref.lower()
    if any(k in ref_l for k in ("sulfide-clean", "halides.csv", "keggin",
                                "literature_additions", "experimental-summary")):
        return "curated_lit"
    if "liverpool" in ref_l:
        return "db_liverpool"
    if "caltech" in ref_l:
        return "db_caltech"
    return "db_obelix"


def _temperature(note: str) -> float | None:
    m = re.search(r"measured_temperature_C=([^;]+)", note)
    if m:
        try:
            return float(m.group(1).strip())
        except ValueError:
            pass
    return None


def _merge_dois(primary: str, extras: list[str]) -> str:
    """Semicolon list: primary DOI first, then unique extras in order."""
    seen: set[str] = set()
    result: list[str] = []
    for doi in [primary] + extras:
        doi = doi.strip()
        if doi and doi not in seen:
            seen.add(doi)
            result.append(doi)
    return "; ".join(result)


def main() -> None:
    with IN_PATH.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        orig_header = list(reader.fieldnames or [])
        rows = list(reader)

    excluded: list[dict] = []
    eligible: list[dict] = []

    for i, r in enumerate(rows):
        cond_raw, _ = parse_conductivity(r["Ionic conductivity (S cm-1)"])
        try:
            cond = float(cond_raw)
        except (TypeError, ValueError):
            cond = float("nan")

        r["_cond"] = cond
        r["_idx"] = i
        r["_is_exp"] = r["ID"].startswith("exp_")
        r["_bucket"] = _source_bucket(r["ID"], r.get("Ref", ""))
        r["_bucket_rank"] = BUCKET_RANK.get(r["_bucket"], 99)
        r["_checked"] = r.get("Checked", "").strip().lower() in ("1", "yes", "true")
        r["Family"] = normalize_family(r.get("Family", ""))

        # formula relabel
        red = r.get("Reduced Composition", "")
        if red in FORMULA_FAMILY_RELABEL:
            r["Family"] = FORMULA_FAMILY_RELABEL[red]

        reason = None
        if math.isnan(cond) or cond <= 0:
            reason = "invalid or non-positive conductivity"
        elif not r["_is_exp"] and cond < MIN_CONDUCTIVITY:
            reason = f"below threshold {MIN_CONDUCTIVITY}"
        else:
            temp = _temperature(r.get("note", ""))
            src_l = r.get("Ref", "").lower()
            is_temp_filtered = "liverpool" in src_l or "sulfide" in src_l
            is_temp_required = "liverpool" in src_l
            if is_temp_filtered and temp is not None:
                if not (TEMP_MIN <= temp <= TEMP_MAX):
                    reason = f"temperature {temp}°C outside [{TEMP_MIN},{TEMP_MAX}]"
            elif is_temp_required and temp is None:
                reason = "Liverpool row: temperature unknown"

        if reason:
            r["_reason"] = reason
            excluded.append(r)
        else:
            eligible.append(r)

    # --- per-Reduced-Composition selection ---
    from collections import defaultdict
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in eligible:
        groups[r["Reduced Composition"]].append(r)

    kept_rows: list[dict] = []
    dup_dropped: list[dict] = []

    for formula, group in sorted(groups.items()):
        if len(group) == 1:
            kept_rows.append(group[0])
            continue

        preferred_family = FORMULA_PREFERRED_FAMILY.get(formula, "")
        exp_rows = [r for r in group if r["_is_exp"]]
        lit_rows = [r for r in group if not r["_is_exp"]]

        # experimental takes absolute priority
        pool = exp_rows if exp_rows else (
            [r for r in lit_rows if r["Family"] == preferred_family] or lit_rows
            if preferred_family else lit_rows
        )

        # sort: highest conductivity, then checked, then bucket rank, then row order
        pool_sorted = sorted(
            pool,
            key=lambda r: (-r["_cond"], not r["_checked"], r["_bucket_rank"], r["_idx"]),
        )
        winner = pool_sorted[0]
        rest = pool_sorted[1:]

        # collect DOIs from rows with identical conductivity (same measurement)
        same_cond = [r for r in rest if r["_cond"] == winner["_cond"]]
        different_cond = [r for r in rest if r["_cond"] != winner["_cond"]]

        # merge DOIs: winner + all others (same-cond replication + different-cond literature)
        extra_dois = [r.get("DOI", "") for r in rest]
        winner["_all_dois"] = _merge_dois(winner.get("DOI", ""), extra_dois)

        kept_rows.append(winner)
        # same-cond rows are merged silently; different-cond are logged as dup-dropped
        for r in same_cond:
            r["_reason"] = f"same conductivity {r['_cond']:.3e} as kept {winner['ID']}; DOI merged"
        for r in different_cond:
            r["_reason"] = f"lower conductivity {r['_cond']:.3e}; kept {winner['ID']} ({winner['_cond']:.3e})"
        dup_dropped.extend(rest)

    # restore original row order
    kept_rows.sort(key=lambda r: r["_idx"])

    # --- build output ---
    out_header = orig_header + ["source_bucket", "all_dois"]
    internal = {k for k in kept_rows[0] if k.startswith("_")} if kept_rows else set()

    def _clean(r: dict) -> dict:
        out = {k: r.get(k, "") for k in out_header}
        out["source_bucket"] = r.get("_bucket", "")
        out["all_dois"] = r.get("_all_dois", r.get("DOI", ""))
        return out

    with OUT_PATH.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=out_header)
        w.writeheader()
        w.writerows(_clean(r) for r in kept_rows)

    excl_header = orig_header + ["source_bucket", "removed_reason"]
    all_excl = excluded + dup_dropped
    all_excl.sort(key=lambda r: r["_idx"])
    with OUT_EXCL.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=excl_header, extrasaction="ignore")
        w.writeheader()
        for r in all_excl:
            row = {k: r.get(k, "") for k in excl_header}
            row["source_bucket"] = r.get("_bucket", "")
            row["removed_reason"] = r.get("_reason", "")
            w.writerow(row)

    # --- summary ---
    bucket_counts = {}
    for r in kept_rows:
        b = r["_bucket"]
        bucket_counts[b] = bucket_counts.get(b, 0) + 1

    family_counts = {}
    for r in kept_rows:
        f = r["Family"] or "(unknown)"
        family_counts[f] = family_counts.get(f, 0) + 1
    top_families = dict(sorted(family_counts.items(), key=lambda x: -x[1])[:15])

    summary = {
        "input_rows": len(rows),
        "eligible_rows": len(eligible),
        "output_rows": len(kept_rows),
        "unique_formulas": len(set(r["Reduced Composition"] for r in kept_rows)),
        "removed_below_threshold": sum(1 for r in excluded if "below threshold" in r.get("_reason","")),
        "removed_invalid": sum(1 for r in excluded if "invalid" in r.get("_reason","")),
        "removed_temperature": sum(1 for r in excluded if "temperature" in r.get("_reason","")),
        "dup_rows_dropped": len(dup_dropped),
        "exp_rows_in_output": sum(1 for r in kept_rows if r["_is_exp"]),
        "source_bucket_counts": bucket_counts,
        "top_family_counts": top_families,
    }
    OUT_SUM.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nOutput  : {OUT_PATH}")
    print(f"Excluded: {OUT_EXCL}")


if __name__ == "__main__":
    main()
