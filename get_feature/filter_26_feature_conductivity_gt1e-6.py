from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parent
FEATURE_DIR = ROOT / "features"
DEFAULT_TRAIN = FEATURE_DIR / "ionic_26_features_random_clipped_1e-10_train.csv"
DEFAULT_TEST = FEATURE_DIR / "ionic_26_features_random_clipped_1e-10_test.csv"
DEFAULT_TRAIN_OUTPUT = FEATURE_DIR / "ionic_26_features_random_gt1e-8_train.csv"
DEFAULT_TEST_OUTPUT = FEATURE_DIR / "ionic_26_features_random_gt1e-8_test.csv"
TARGET_COLUMN = "Ionic conductivity (S cm-1)"


def read_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames
        if not fieldnames or TARGET_COLUMN not in fieldnames:
            raise ValueError(f"{path} does not contain {TARGET_COLUMN}")
        return list(reader), fieldnames


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_conductivity(value: str) -> tuple[float | None, bool]:
    text = str(value).strip().replace("−", "-")
    is_upper_bound = text.startswith(("<", "≤"))
    numeric_text = text.lstrip("<>≤≥=~ ").replace(",", "")
    try:
        return float(numeric_text), is_upper_bound
    except ValueError:
        return None, is_upper_bound


def keep_row(row: dict[str, str], min_conductivity: float) -> bool:
    value, is_upper_bound = parse_conductivity(row[TARGET_COLUMN])
    if value is None:
        return True
    if value < min_conductivity:
        return False
    if is_upper_bound and value <= min_conductivity:
        return False
    return True


def filter_rows(
    rows: list[dict[str, str]], min_conductivity: float
) -> tuple[list[dict[str, str]], int, int]:
    kept_rows = [row for row in rows if keep_row(row, min_conductivity)]
    removed_count = len(rows) - len(kept_rows)
    return kept_rows, removed_count, len(rows)


def filter_file(input_path: Path, output_path: Path, min_conductivity: float) -> int:
    rows, fieldnames = read_rows(input_path)
    kept_rows, removed_count, original_count = filter_rows(rows, min_conductivity)
    write_rows(output_path, fieldnames, kept_rows)
    print(
        f"{input_path} -> {output_path}: "
        f"rows={original_count}, kept={len(kept_rows)}, removed={removed_count}"
    )
    return removed_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Remove rows with low ionic conductivity while preserving the existing "
            "random train/test split."
        )
    )
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--train-output", type=Path, default=DEFAULT_TRAIN_OUTPUT)
    parser.add_argument("--test-output", type=Path, default=DEFAULT_TEST_OUTPUT)
    parser.add_argument("--min-conductivity", type=float, default=1e-8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_removed = filter_file(args.train, args.train_output, args.min_conductivity)
    test_removed = filter_file(args.test, args.test_output, args.min_conductivity)
    print(f"Minimum conductivity: {args.min_conductivity:.12g}")
    print(f"Total removed rows: {train_removed + test_removed}")


if __name__ == "__main__":
    main()
