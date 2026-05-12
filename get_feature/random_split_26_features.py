from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RAWDATA = ROOT / "rawdata"
FEATURE_DIR = ROOT / "features"
DEFAULT_FEATURES = FEATURE_DIR / "ionic_26_features_all.csv"
DEFAULT_RAW_ALL = RAWDATA / "all.csv"
DEFAULT_RAW_TRAIN = RAWDATA / "random_train.csv"
DEFAULT_RAW_TEST = RAWDATA / "random_test.csv"
DEFAULT_FEATURE_TRAIN = FEATURE_DIR / "ionic_26_features_random_train.csv"
DEFAULT_FEATURE_TEST = FEATURE_DIR / "ionic_26_features_random_test.csv"
TARGET_COLUMN = "Ionic conductivity (S cm-1)"


def read_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames
        if not fieldnames or "ID" not in fieldnames:
            raise ValueError(f"{path} does not contain an ID column")
        return list(reader), fieldnames


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def row_id(row: dict[str, str]) -> str:
    return row.get("ID", "").strip()


def parse_conductivity(value: str) -> tuple[float | None, bool]:
    text = str(value).strip().replace("−", "-")
    is_upper_bound = text.startswith(("<", "≤"))
    numeric_text = text.lstrip("<>≤≥=~ ").replace(",", "")
    try:
        return float(numeric_text), is_upper_bound
    except ValueError:
        return None, is_upper_bound


def clip_conductivity_rows(rows: list[dict[str, str]], min_conductivity: float | None) -> tuple[list[dict[str, str]], int]:
    if min_conductivity is None:
        return [dict(row) for row in rows], 0

    clipped_rows = []
    clipped_count = 0
    clipped_value = f"{min_conductivity:.12g}"
    for row in rows:
        next_row = dict(row)
        value, is_upper_bound = parse_conductivity(next_row.get(TARGET_COLUMN, ""))
        if value is not None and (value < min_conductivity or (is_upper_bound and value <= min_conductivity)):
            next_row[TARGET_COLUMN] = clipped_value
            clipped_count += 1
        clipped_rows.append(next_row)
    return clipped_rows, clipped_count


def validate_unique_ids(rows: list[dict[str, str]], path: Path) -> list[str]:
    ids = [row_id(row) for row in rows if row_id(row)]
    seen: set[str] = set()
    duplicates: set[str] = set()
    for id_ in ids:
        if id_ in seen:
            duplicates.add(id_)
        seen.add(id_)
    if duplicates:
        raise ValueError(f"Duplicate IDs in {path}: {sorted(duplicates)[:10]}")
    return ids


def split_ids(ids: list[str], test_size: float, seed: int) -> tuple[set[str], set[str]]:
    if not 0 < test_size < 1:
        raise ValueError("--test-size must be between 0 and 1")
    shuffled = ids[:]
    random.Random(seed).shuffle(shuffled)
    test_count = max(1, min(len(shuffled) - 1, round(len(shuffled) * test_size)))
    test_ids = set(shuffled[:test_count])
    train_ids = set(shuffled[test_count:])
    return train_ids, test_ids


def select_rows(rows: list[dict[str, str]], ids: set[str]) -> list[dict[str, str]]:
    return [row for row in rows if row_id(row) in ids]


def random_split(
    features_path: Path,
    raw_all_path: Path,
    feature_train_output: Path,
    feature_test_output: Path,
    raw_train_output: Path,
    raw_test_output: Path,
    test_size: float,
    seed: int,
    min_conductivity: float | None,
) -> None:
    feature_rows, feature_fieldnames = read_rows(features_path)
    feature_ids = validate_unique_ids(feature_rows, features_path)
    train_ids, test_ids = split_ids(feature_ids, test_size, seed)

    train_feature_rows, train_feature_clipped = clip_conductivity_rows(select_rows(feature_rows, train_ids), min_conductivity)
    test_feature_rows, test_feature_clipped = clip_conductivity_rows(select_rows(feature_rows, test_ids), min_conductivity)
    write_rows(feature_train_output, feature_fieldnames, train_feature_rows)
    write_rows(feature_test_output, feature_fieldnames, test_feature_rows)

    raw_rows, raw_fieldnames = read_rows(raw_all_path)
    validate_unique_ids(raw_rows, raw_all_path)
    raw_id_set = {row_id(row) for row in raw_rows}
    missing_raw = sorted((train_ids | test_ids) - raw_id_set)

    train_raw_rows, train_raw_clipped = clip_conductivity_rows(select_rows(raw_rows, train_ids), min_conductivity)
    test_raw_rows, test_raw_clipped = clip_conductivity_rows(select_rows(raw_rows, test_ids), min_conductivity)
    write_rows(raw_train_output, raw_fieldnames, train_raw_rows)
    write_rows(raw_test_output, raw_fieldnames, test_raw_rows)

    print(f"Feature source: {features_path}")
    print(f"Random seed: {seed}")
    print(f"Test size: {test_size:.2f}")
    if min_conductivity is not None:
        print(f"Minimum conductivity: {min_conductivity:.12g}")
    print(f"Wrote train features: {len(train_feature_rows)} -> {feature_train_output}")
    print(f"Wrote test features: {len(test_feature_rows)} -> {feature_test_output}")
    print(f"Wrote raw train rows: {len(train_raw_rows)} -> {raw_train_output}")
    print(f"Wrote raw test rows: {len(test_raw_rows)} -> {raw_test_output}")
    if min_conductivity is not None:
        print(f"Clipped feature rows: train={train_feature_clipped}, test={test_feature_clipped}")
        print(f"Clipped raw rows: train={train_raw_clipped}, test={test_raw_clipped}")
    if missing_raw:
        print(f"Warning: {len(missing_raw)} selected feature IDs are missing in raw all.csv: {missing_raw[:10]}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a reproducible random train/test split for the 26-feature dataset."
    )
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--raw-all", type=Path, default=DEFAULT_RAW_ALL)
    parser.add_argument("--feature-train-output", type=Path, default=DEFAULT_FEATURE_TRAIN)
    parser.add_argument("--feature-test-output", type=Path, default=DEFAULT_FEATURE_TEST)
    parser.add_argument("--raw-train-output", type=Path, default=DEFAULT_RAW_TRAIN)
    parser.add_argument("--raw-test-output", type=Path, default=DEFAULT_RAW_TEST)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-conductivity", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random_split(
        features_path=args.features,
        raw_all_path=args.raw_all,
        feature_train_output=args.feature_train_output,
        feature_test_output=args.feature_test_output,
        raw_train_output=args.raw_train_output,
        raw_test_output=args.raw_test_output,
        test_size=args.test_size,
        seed=args.seed,
        min_conductivity=args.min_conductivity,
    )


if __name__ == "__main__":
    main()
