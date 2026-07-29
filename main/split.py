from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_PATH = ROOT / "features" / "ionic_main_random_gt1e-6_train.csv"
DEFAULT_TEST_PATH = ROOT / "features" / "ionic_main_random_gt1e-6_test.csv"


@dataclass
class SplitConfig:
    """Options for train/test splitting.

    method:
        Split strategy. Supported values:
        - "random": reproducible random split over all rows.
        - "fixed_by_id": use explicit train_ids and test_ids.
        - "family_stratified": random split stratified by Family or
          stratify_column.
        - "group": split by group_column so the same group does not appear in
          both train and test.
        - "argyrodite_only": keep only argyrodite-family rows, then random
          split that subset.
    test_size:
        Fraction assigned to the test set for random-like split methods.
    seed:
        Random seed for reproducible split methods.
    group_column:
        Required when method="group"; examples include Family, Ref, DOI, or a
        parent-formula column.
    stratify_column:
        Optional column for method="family_stratified"; defaults to Family.
    train_ids:
        Required when method="fixed_by_id"; IDs assigned to train.
    test_ids:
        Required when method="fixed_by_id"; IDs assigned to test.
    train_output:
        Train CSV path. Use None to skip automatic file writing.
    test_output:
        Test CSV path. Use None to skip automatic file writing.
    """

    method: str = "random"
    test_size: float = 0.2
    seed: int = 42
    group_column: str | None = None
    stratify_column: str | None = None
    train_ids: set[str] | None = None
    test_ids: set[str] | None = None
    train_output: Path | None = DEFAULT_TRAIN_PATH
    test_output: Path | None = DEFAULT_TEST_PATH


@dataclass
class SplitResult:
    train: pd.DataFrame
    test: pd.DataFrame
    summary: dict
    train_path: Path | None = None
    test_path: Path | None = None

    def to_files(self, train_path: str | Path | None = None, test_path: str | Path | None = None) -> None:
        train_output = Path(train_path or self.train_path or DEFAULT_TRAIN_PATH)
        test_output = Path(test_path or self.test_path or DEFAULT_TEST_PATH)
        train_output.parent.mkdir(parents=True, exist_ok=True)
        test_output.parent.mkdir(parents=True, exist_ok=True)
        self.train.to_csv(train_output, index=False)
        self.test.to_csv(test_output, index=False)
        self.train_path = train_output
        self.test_path = test_output


def _distribution(frame: pd.DataFrame, column: str) -> dict:
    if column not in frame.columns:
        return {}
    return frame[column].astype(str).value_counts(dropna=False).to_dict()


def _target_summary(frame: pd.DataFrame) -> dict:
    if "log10_conductivity" not in frame.columns or frame.empty:
        return {}
    values = frame["log10_conductivity"]
    return {
        "min": float(values.min()),
        "median": float(values.median()),
        "max": float(values.max()),
        "mean": float(values.mean()),
    }


def split_feature_table(
    df: pd.DataFrame,
    config: SplitConfig | None = None,
) -> SplitResult:
    """Split a feature table and optionally save train/test CSV files."""

    config = config or SplitConfig()
    method = config.method.lower()
    data = df.reset_index(drop=True).copy()

    if method == "fixed_by_id":
        if config.train_ids is None or config.test_ids is None:
            raise ValueError("fixed_by_id split requires train_ids and test_ids")
        train_ids = {str(item) for item in config.train_ids} #集合行列式，自动去重
        test_ids = {str(item) for item in config.test_ids}
        train = data[data["ID"].astype(str).isin(train_ids)].copy()
        test = data[data["ID"].astype(str).isin(test_ids)].copy()
    elif method == "family_stratified":
        column = config.stratify_column or "Family"
        stratify = data[column] if column in data.columns else None
        train, test = train_test_split(
            data,
            test_size=config.test_size,
            random_state=config.seed,
            shuffle=True,
            stratify=stratify,
        )
    elif method == "group":
        column = config.group_column
        if column is None or column not in data.columns:
            raise ValueError("group split requires an existing group_column")
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=config.test_size,
            random_state=config.seed,
        )
        train_idx, test_idx = next(splitter.split(data, groups=data[column].astype(str)))
        train = data.iloc[train_idx].copy()
        test = data.iloc[test_idx].copy()
    elif method == "argyrodite_only":
        if "Family" not in data.columns:
            raise ValueError("argyrodite_only split requires Family column")
        subset = data[data["Family"].astype(str).str.contains("argyro", case=False, na=False)].copy()
        train, test = train_test_split(
            subset,
            test_size=config.test_size,
            random_state=config.seed,
            shuffle=True,
        )
    elif method == "random":
        train, test = train_test_split(
            data,
            test_size=config.test_size,
            random_state=config.seed,
            shuffle=True,
        )
    else:
        raise ValueError(f"Unsupported split method: {config.method}")

    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    overlap = set(train["ID"].astype(str)) & set(test["ID"].astype(str)) if "ID" in data.columns else set()
    if overlap:
        raise RuntimeError(f"Train/test ID overlap detected: {sorted(overlap)[:10]}")

    summary = {
        "config": asdict(config),
        "input_rows": int(len(data)),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "target_train": _target_summary(train),
        "target_test": _target_summary(test),
        "family_train": _distribution(train, "Family"),
        "family_test": _distribution(test, "Family"),
    }
    result = SplitResult(
        train=train,
        test=test,
        summary=summary,
        train_path=config.train_output,
        test_path=config.test_output,
    )
    if config.train_output is not None and config.test_output is not None:
        result.to_files(config.train_output, config.test_output)
    return result
