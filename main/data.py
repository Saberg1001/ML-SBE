from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
from pymatgen.core import Composition

from .features import (
    CHARGE_RESIDUAL_EXCLUDE_LIMIT,
    MANUAL_ABNORMAL_CHARGE_IDS,
    charge_balance_records,
    contains_organic_molecule,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_PATH = ROOT / "rawdata" / "all.csv"


@dataclass
class CleanDataConfig:
    """Options for raw OBLiX data loading and row-level cleaning.

    raw_path:
        CSV path used when clean_raw_data receives no dataframe.
    remove_organic:
        If True, remove formulas containing both C and H.
    remove_charge_abnormal:
        If True, remove rows with large oxidation-state charge residuals or
        manually listed abnormal IDs.
    charge_residual_limit:
        Absolute residual-charge threshold used by remove_charge_abnormal.
    """

    raw_path: Path = DEFAULT_RAW_PATH
    remove_organic: bool = True
    remove_charge_abnormal: bool = True
    charge_residual_limit: float = CHARGE_RESIDUAL_EXCLUDE_LIMIT


@dataclass
class CleanDataResult:
    cleaned: pd.DataFrame
    removed: pd.DataFrame
    summary: dict

    def to_files(self, cleaned_path: str | Path, removed_path: str | Path) -> None:
        Path(cleaned_path).parent.mkdir(parents=True, exist_ok=True)
        Path(removed_path).parent.mkdir(parents=True, exist_ok=True)
        self.cleaned.to_csv(cleaned_path, index=False)
        self.removed.to_csv(removed_path, index=False)


def load_raw_data(path: str | Path = DEFAULT_RAW_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def _organic_mask(frame: pd.DataFrame) -> pd.Series:
    def is_organic(formula: object) -> bool:
        try:
            return contains_organic_molecule(Composition(str(formula)))
        except Exception:
            return False

    return frame["True Composition"].apply(is_organic)


def _charge_abnormal_ids(frame: pd.DataFrame, limit: float) -> tuple[set[str], pd.DataFrame]:
    records = pd.DataFrame(charge_balance_records(frame))
    if records.empty:
        return set(), records

    residual = records["residual_charge"].astype(float)
    abnormal = records[
        (residual.abs() > limit)
        | records["ID"].astype(str).isin(MANUAL_ABNORMAL_CHARGE_IDS)
    ].copy()
    reasons = []
    for _, row in abnormal.iterrows():
        row_reasons = []
        if abs(float(row["residual_charge"])) > limit:
            row_reasons.append(f"abs(residual_charge) > {limit:g}")
        if str(row["ID"]) in MANUAL_ABNORMAL_CHARGE_IDS:
            row_reasons.append("manual abnormal charge screening")
        reasons.append("; ".join(row_reasons))
    if not abnormal.empty:
        abnormal["removed_reason"] = reasons
    return set(abnormal["ID"].astype(str)), abnormal


def clean_raw_data(
    df: pd.DataFrame | None = None,
    config: CleanDataConfig | None = None,
) -> CleanDataResult:
    """Load or receive raw data, remove configured abnormal rows, and report removals."""

    config = config or CleanDataConfig()
    raw = load_raw_data(config.raw_path) if df is None else df.copy()
    cleaned = raw.copy()
    removed_frames = []

    if config.remove_organic:
        organic = _organic_mask(cleaned)
        removed = cleaned[organic].copy()
        if not removed.empty:
            removed["removed_reason"] = "organic-like formula"
            removed_frames.append(removed)
        cleaned = cleaned[~organic].copy()

    charge_removed_count = 0
    if config.remove_charge_abnormal:
        abnormal_ids, abnormal_records = _charge_abnormal_ids(
            cleaned,
            config.charge_residual_limit,
        )
        if abnormal_ids:
            charge_removed = cleaned[cleaned["ID"].astype(str).isin(abnormal_ids)].copy()
            reason_by_id = abnormal_records.set_index("ID")["removed_reason"]
            charge_removed["removed_reason"] = charge_removed["ID"].map(reason_by_id)
            removed_frames.append(charge_removed)
            cleaned = cleaned[~cleaned["ID"].astype(str).isin(abnormal_ids)].copy()
            charge_removed_count = len(charge_removed)

    removed_all = (
        pd.concat(removed_frames, ignore_index=True)
        if removed_frames
        else pd.DataFrame(columns=[*raw.columns, "removed_reason"])
    )
    cleaned = cleaned.reset_index(drop=True)

    summary = {
        "config": asdict(config),
        "raw_rows": int(len(raw)),
        "cleaned_rows": int(len(cleaned)),
        "removed_rows": int(len(removed_all)),
        "removed_organic_rows": int(
            (removed_all.get("removed_reason", pd.Series(dtype=str)) == "organic-like formula").sum()
        ),
        "removed_charge_abnormal_rows": int(charge_removed_count),
    }
    return CleanDataResult(cleaned=cleaned, removed=removed_all, summary=summary)
