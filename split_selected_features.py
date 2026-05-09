from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RAWDATA = ROOT / "rawdata"
FEATURE_DIR = ROOT / "features"
DEFAULT_FEATURES = FEATURE_DIR / "ionic_26_features_all.csv"
DEFAULT_ALL = RAWDATA / "all.csv"
DEFAULT_TRAIN = RAWDATA / "train.csv"
DEFAULT_TEST = RAWDATA / "test.csv"
DEFAULT_FEATURE_TRAIN = FEATURE_DIR / "ionic_26_features_train.csv"
DEFAULT_FEATURE_TEST = FEATURE_DIR / "ionic_26_features_test.csv"


def read_rows(path: Path, *, required: bool = True) -> tuple[list[dict[str, str]], list[str]] | None:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            fieldnames = reader.fieldnames
            if not fieldnames or "ID" not in fieldnames:
                raise ValueError(f"{path} does not contain an ID column")
            return list(reader), fieldnames
    except (UnicodeDecodeError, csv.Error, ValueError) as exc:
        if required:
            raise RuntimeError(f"Failed to read CSV {path}: {exc}") from exc
        print(f"Warning: could not read {path} as CSV ({exc}); rebuilding train IDs from all.csv.")
        return None


def read_id_set(path: Path, *, required: bool = True) -> set[str] | None:
    loaded = read_rows(path, required=required)
    if loaded is None:
        return None
    rows, _ = loaded
    return {row["ID"].strip() for row in rows if row.get("ID", "").strip()}


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def split_features(
    features_path: Path,
    all_path: Path,
    train_path: Path,
    test_path: Path,
    train_output: Path,
    test_output: Path,
) -> None:
    feature_rows, fieldnames = read_rows(features_path)
    feature_ids = [row["ID"].strip() for row in feature_rows]
    duplicated = sorted({id_ for id_ in feature_ids if feature_ids.count(id_) > 1})
    if duplicated:
        raise RuntimeError(f"Duplicate IDs in {features_path}: {duplicated[:10]}")

    test_ids = read_id_set(test_path)
    train_ids = read_id_set(train_path, required=False)
    source = train_path.name
    if train_ids is None:
        all_ids = read_id_set(all_path)
        train_ids = all_ids - test_ids
        source = f"{all_path.name} minus {test_path.name}"

    overlap = train_ids & test_ids
    if overlap:
        raise RuntimeError(f"Train/test ID overlap detected: {sorted(overlap)[:10]}")

    feature_id_set = set(feature_ids)
    missing_train = sorted(train_ids - feature_id_set)
    missing_test = sorted(test_ids - feature_id_set)

    train_rows = [row for row in feature_rows if row["ID"].strip() in train_ids]
    test_rows = [row for row in feature_rows if row["ID"].strip() in test_ids]
    unassigned = sorted(feature_id_set - train_ids - test_ids)

    write_rows(train_output, fieldnames, train_rows)
    write_rows(test_output, fieldnames, test_rows)

    print(f"Feature rows: {len(feature_rows)}")
    print(f"Train source: {source}")
    print(f"Wrote train features: {len(train_rows)} -> {train_output}")
    print(f"Wrote test features: {len(test_rows)} -> {test_output}")
    if missing_train:
        print(f"Warning: {len(missing_train)} train IDs have no selected features: {missing_train[:10]}")
    if missing_test:
        print(f"Warning: {len(missing_test)} test IDs have no selected features: {missing_test[:10]}")
    if unassigned:
        print(f"Warning: {len(unassigned)} feature rows were not assigned: {unassigned[:10]}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split selected feature rows into train/test feature CSV files."
    )
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--all", type=Path, default=DEFAULT_ALL)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--train-output", type=Path, default=DEFAULT_FEATURE_TRAIN)
    parser.add_argument("--test-output", type=Path, default=DEFAULT_FEATURE_TEST)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_features(
        features_path=args.features,
        all_path=args.all,
        train_path=args.train,
        test_path=args.test,
        train_output=args.train_output,
        test_output=args.test_output,
    )


if __name__ == "__main__":
    main()
