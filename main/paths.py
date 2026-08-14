from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_DIR = PROJECT_ROOT / "config"
CHEMISTRY_CONFIG_DIR = CONFIG_DIR / "chemistry"

DATA_DIR = PROJECT_ROOT / "data"
OBELIX_RAW_DIR = DATA_DIR / "obelix" / "raw"
TREND_DIR = DATA_DIR / "trend"
EXPERIMENTAL_DIR = DATA_DIR / "experimental"
EXPERIMENTAL_RAW_DIR = EXPERIMENTAL_DIR / "raw"
EXPERIMENTAL_ANNOTATIONS_DIR = EXPERIMENTAL_DIR / "annotations"
PREDICTION_INPUTS_DIR = DATA_DIR / "prediction_inputs"
MODELING_DIR = DATA_DIR / "modeling"
SPLITS_DIR = DATA_DIR / "splits"

RUNS_DIR = PROJECT_ROOT / "runs"
REPORTS_DIR = PROJECT_ROOT / "reports"
ARCHIVE_DIR = PROJECT_ROOT / "archive"


def portable_path(path: str | Path) -> str:
    """Return a project-relative path when possible."""
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)
