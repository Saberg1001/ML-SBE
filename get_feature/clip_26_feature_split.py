from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parent
FEATURE_DIR = ROOT / "features"
DEFAULT_TRAIN = FEATURE_DIR / "ionic_26_features_train.csv"
DEFAULT_TEST = FEATURE_DIR / "ionic_26_features_test.csv"
DEFAULT_TRAIN_OUTPUT = FEATURE_DIR / "ionic_26_features_clipped_1e-10_train.csv"
DEFAULT_TEST_OUTPUT = FEATURE_DIR / "ionic_26_features_clipped_1e-10_test.csv"
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


def clip_rows(rows: list[dict[str, str]], min_conductivity: float) -> tuple[list[dict[str, str]], int]:
    clipped = []
    clipped_count = 0
    clipped_value = f"{min_conductivity:.12g}"
    for row in rows:
        next_row = dict(row)
        value, is_upper_bound = parse_conductivity(next_row[TARGET_COLUMN])
        if value is not None and (value < min_conductivity or (is_upper_bound and value <= min_conductivity)):
            next_row[TARGET_COLUMN] = clipped_value
            clipped_count += 1
        clipped.append(next_row)
    return clipped, clipped_count


def clip_file(input_path: Path, output_path: Path, min_conductivity: float) -> int:
    rows, fieldnames = read_rows(input_path)
    clipped_rows, clipped_count = clip_rows(rows, min_conductivity)
    write_rows(output_path, fieldnames, clipped_rows)
    print(f"{input_path} -> {output_path}: rows={len(rows)}, clipped={clipped_count}")
    return clipped_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clip low conductivity values while preserving the existing 26-feature train/test split."
    )
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--train-output", type=Path, default=DEFAULT_TRAIN_OUTPUT)
    parser.add_argument("--test-output", type=Path, default=DEFAULT_TEST_OUTPUT)
    parser.add_argument("--min-conductivity", type=float, default=1e-10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_clipped = clip_file(args.train, args.train_output, args.min_conductivity)
    test_clipped = clip_file(args.test, args.test_output, args.min_conductivity)
    print(f"Minimum conductivity: {args.min_conductivity:.12g}")
    print(f"Total clipped rows: {train_clipped + test_clipped}")


if __name__ == "__main__":
    main()
