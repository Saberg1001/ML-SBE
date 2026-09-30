"""Patch family labels in data-absolute-v3.csv based on manual review.

Corrections applied
-------------------
1. chlorides (14 rows) → halides
2. Liverpool perovskite series (5 rows in "other") → perovskites
3. Experimental unknown argyrodites (40 rows) → argyrodites
4. Single-entry mislabels:
   - layered (opo Li0.8Sn0.8S2) → sulfides
   - hexaoxometalates (jqc Li7BiO6) → oxides
   - borophosphates (caltech_icsd_193168) → phosphates
   - zircon (liverpool_2000_0790 Li0.6Y0.8PO4) → phosphates
5. Remove hybrid_halide row (trv1_0149 ZnH12C4(Br2N)2, no Li)

Usage
-----
    python main/absolute/patch_family_v3.py
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

if __package__ is None:
    _ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)

from main.paths import DATA_DIR

ABSOLUTE_DIR = DATA_DIR / "absolute"
IN_PATH = ABSOLUTE_DIR / "data-absolute-v3.csv"
OUT_PATH = ABSOLUTE_DIR / "data-absolute-v3.csv"
BACKUP_PATH = ABSOLUTE_DIR / "data-absolute-v3.csv.backup"


def _is_experimental_argyrodite(row_id: str, family: str, formula: str) -> bool:
    """Experimental unknown rows that are actually argyrodites."""
    if not row_id.startswith("exp_"):
        return False
    if family != "unknown":
        return False
    # Li5.5+/Li6+ with (P/As/Sb/Si) + S + (Br/I/Cl)
    formula_l = formula.lower()
    has_li = "li" in formula_l
    has_chalco = any(x in formula_l for x in ("as", "sb", "si", "p"))
    has_s = "s" in formula_l
    has_halogen = any(x in formula_l for x in ("br", "i", "cl"))
    return has_li and has_chalco and has_s and has_halogen


def main() -> None:
    # Backup original
    if not BACKUP_PATH.exists():
        import shutil
        shutil.copy(IN_PATH, BACKUP_PATH)
        print(f"Backup saved: {BACKUP_PATH}")

    with IN_PATH.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        header = list(reader.fieldnames or [])
        rows = list(reader)

    changes: list[dict] = []
    removed_rows: list[dict] = []

    for r in rows:
        row_id = r["ID"]
        old_family = r["Family"]
        new_family = old_family
        reason = ""

        # Rule 1: chlorides → halides
        if old_family == "chlorides":
            new_family = "halides"
            reason = "chlorides → halides (all are halide SSEs)"

        # Rule 2: Liverpool perovskites
        elif old_family == "other" and "liverpool" in r.get("Ref", "").lower():
            comp = r.get("Reduced Composition", "")
            if "LiLaNbO" in comp or "LiNdNbO" in comp or "LiSmNbO" in comp:
                new_family = "perovskites"
                reason = "Liverpool LiLaNbO3 series → perovskites"

        # Rule 3: Experimental unknown argyrodites
        elif _is_experimental_argyrodite(row_id, old_family, r.get("Reduced Composition", "")):
            new_family = "argyrodites"
            reason = "experimental Li5.5+PS/AsS/SbS + halogen → argyrodites"

        # Rule 4: Single-entry fixes
        elif old_family == "layered" and row_id == "opo":
            new_family = "sulfides"
            reason = "layered Li0.8Sn0.8S2 → sulfides"
        elif old_family == "hexaoxometalates" and row_id == "jqc":
            new_family = "oxides"
            reason = "hexaoxometalates Li7BiO6 → oxides"
        elif old_family == "borophosphates" and row_id == "caltech_icsd_193168":
            new_family = "phosphates"
            reason = "borophosphates → phosphates"
        elif old_family == "zircon" and row_id == "liverpool_2000_0790":
            new_family = "phosphates"
            reason = "zircon Li0.6Y0.8PO4 → phosphates"

        # Rule 5: Remove non-Li row
        if old_family == "hybrid_halide" and row_id == "trv1_0149":
            removed_rows.append(r)
            reason = "REMOVED: ZnH12C4(Br2N)2 contains no Li"
            changes.append({
                "ID": row_id,
                "old_family": old_family,
                "new_family": "[REMOVED]",
                "reason": reason,
            })
            continue

        if new_family != old_family:
            r["Family"] = new_family
            changes.append({
                "ID": row_id,
                "old_family": old_family,
                "new_family": new_family,
                "reason": reason,
            })

    # Write corrected data (exclude removed rows)
    kept_rows = [r for r in rows if r not in removed_rows]
    with OUT_PATH.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=header)
        w.writeheader()
        w.writerows(kept_rows)

    # Report
    print(f"\n✓ Patched {len(changes)} rows")
    print(f"  - Removed: {len(removed_rows)}")
    print(f"  - Relabeled: {len(changes) - len(removed_rows)}")
    print(f"\nOutput: {OUT_PATH}")
    print(f"Backup: {BACKUP_PATH}\n")

    if changes:
        print("Changes summary:")
        for c in changes[:10]:
            print(f"  {c['ID']:20s}  {c['old_family']:20s} → {c['new_family']:20s}  ({c['reason']})")
        if len(changes) > 10:
            print(f"  ... and {len(changes) - 10} more")


if __name__ == "__main__":
    main()
