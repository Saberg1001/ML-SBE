from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parent
FEATURE_DIR = ROOT / "features"
TARGET_COLUMN = "Ionic conductivity (S cm-1)"

DEFAULT_TRAIN_INPUT = FEATURE_DIR / "ionic_26_features_random_gt1e-6_train.csv"
DEFAULT_TEST_INPUT = FEATURE_DIR / "ionic_26_features_random_gt1e-6_test.csv"
DEFAULT_TRAIN_OUTPUT = FEATURE_DIR / "ionic_8_features_random_gt1e-6_train.csv"
DEFAULT_TEST_OUTPUT = FEATURE_DIR / "ionic_8_features_random_gt1e-6_test.csv"

ID_COLUMNS = ["ID", "True Composition", "Z_by_element"]
SELECTED_FEATURES = [
    "n_Li",
    "χ⁻",
    "χ⁺(excl Li⁺)",
    "r⁺(excl Li⁺) (pm)",
    "r⁻ (pm)",
    "nₕₒₛₜ cₐₜᵢₒₙ",
    "ρ⁺(excl Li⁺) (C m⁻³)",
    "Φ⁺(excl Li⁺) (|Z| pm⁻¹)",
]
EXCL_LI_FEATURES = [
    "χ⁺(excl Li⁺)",
    "r⁺(excl Li⁺) (pm)",
    "ρ⁺(excl Li⁺) (C m⁻³)",
    "Φ⁺(excl Li⁺) (|Z| pm⁻¹)",
]


def parse_float(value: str, column: str, path: Path, row_index: int) -> float:
    try:
        return float(value)
    except ValueError as exc:
        raise RuntimeError(
            f"Could not parse {column!r} as float in {path} row {row_index}: {value!r}"
        ) from exc


def fill_missing_excl_li_features(selected: dict[str, str], path: Path, row_index: int) -> None:
    host_fraction = parse_float(
        selected["nₕₒₛₜ cₐₜᵢₒₙ"],
        "nₕₒₛₜ cₐₜᵢₒₙ",
        path,
        row_index,
    )
    if host_fraction != 0:
        return

    for column in EXCL_LI_FEATURES:
        value = selected[column].strip()
        if not value or value.lower() == "nan":
            selected[column] = "0"


def write_selected(input_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_columns = [*ID_COLUMNS, *SELECTED_FEATURES, TARGET_COLUMN]

    with input_path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames:
            raise RuntimeError(f"{input_path} has no header")

        required_columns = {*ID_COLUMNS, *SELECTED_FEATURES, TARGET_COLUMN}
        missing = sorted(required_columns - set(reader.fieldnames))
        if missing:
            raise RuntimeError(f"{input_path} is missing required columns: {missing}")

        rows = []
        for row_index, row in enumerate(reader, start=2):
            selected = {column: row[column] for column in output_columns}
            fill_missing_excl_li_features(selected, input_path, row_index)
            rows.append(selected)

    with output_path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=output_columns)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows and {len(SELECTED_FEATURES)} features -> {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create the 8-feature gt1e-6 random split.")
    parser.add_argument("--train-input", type=Path, default=DEFAULT_TRAIN_INPUT)
    parser.add_argument("--test-input", type=Path, default=DEFAULT_TEST_INPUT)
    parser.add_argument("--train-output", type=Path, default=DEFAULT_TRAIN_OUTPUT)
    parser.add_argument("--test-output", type=Path, default=DEFAULT_TEST_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_selected(args.train_input, args.train_output)
    write_selected(args.test_input, args.test_output)


if __name__ == "__main__":
    main()
