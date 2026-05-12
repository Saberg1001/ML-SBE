from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent
FEATURE_DIR = ROOT / "features"
RAWDATA_DIR = ROOT / "rawdata"

DEFAULT_TRAIN_INPUT = FEATURE_DIR / "ionic_8_features_random_gt1e-6_train.csv"
DEFAULT_TEST_INPUT = FEATURE_DIR / "ionic_8_features_random_gt1e-6_test.csv"
DEFAULT_RAW = RAWDATA_DIR / "all.csv"


def load_feature_pool(train_input: Path, test_input: Path) -> pd.DataFrame:
    train = pd.read_csv(train_input)
    test = pd.read_csv(test_input)
    pool = pd.concat([train, test], ignore_index=True)
    if pool["ID"].duplicated().any():
        duplicated = pool.loc[pool["ID"].duplicated(), "ID"].tolist()
        raise RuntimeError(f"Duplicated IDs in feature pool: {duplicated[:10]}")
    return pool


def argyrodite_mask(raw: pd.DataFrame, subset: str) -> pd.Series:
    family = raw["Family"].astype(str)
    is_argyrodite = family.str.contains("argyro", case=False, na=False)
    if subset == "argyrodite":
        return is_argyrodite
    composition = raw["Reduced Composition"].astype(str)
    return is_argyrodite & composition.str.contains("S", na=False)


def split_argyrodite_features(
    train_input: Path,
    test_input: Path,
    raw_path: Path,
    train_output: Path,
    test_output: Path,
    sample_output: Path,
    subset: str,
    test_size: float,
    seed: int,
) -> None:
    pool = load_feature_pool(train_input, test_input)
    raw = pd.read_csv(raw_path)
    raw_subset = raw.loc[argyrodite_mask(raw, subset), ["ID", "Reduced Composition", "Family", "Ref"]].copy()

    merged = pool.merge(raw_subset, on="ID", how="inner")
    if merged.empty:
        raise RuntimeError(f"No {subset} samples found in {train_input} + {test_input}")

    feature_columns = pool.columns.tolist()
    train_ids, test_ids = train_test_split(
        merged["ID"],
        test_size=test_size,
        random_state=seed,
        shuffle=True,
    )
    train_ids = set(train_ids)
    test_ids = set(test_ids)

    train = pool[pool["ID"].isin(train_ids)].copy()
    test = pool[pool["ID"].isin(test_ids)].copy()
    samples = merged.assign(split=merged["ID"].map(lambda item: "test" if item in test_ids else "train"))

    train_output.parent.mkdir(parents=True, exist_ok=True)
    sample_output.parent.mkdir(parents=True, exist_ok=True)
    train.to_csv(train_output, index=False)
    test.to_csv(test_output, index=False)
    samples[["split", "ID", "Reduced Composition", "Family", "Ref", *feature_columns]].to_csv(
        sample_output,
        index=False,
    )

    print(f"Subset: {subset}")
    print(f"Samples: {len(merged)}")
    print(f"Train: {len(train)} -> {train_output}")
    print(f"Test: {len(test)} -> {test_output}")
    print(f"Sample list: {sample_output}")
    print("Ref counts:")
    print(samples.groupby(["split", "Ref"]).size().to_string())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split 8-feature argyrodite samples into train/test CSVs.")
    parser.add_argument("--train-input", type=Path, default=DEFAULT_TRAIN_INPUT)
    parser.add_argument("--test-input", type=Path, default=DEFAULT_TEST_INPUT)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--subset", choices=["argyrodite", "s_argyrodite"], default="argyrodite")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--train-output",
        type=Path,
        default=FEATURE_DIR / "ionic_8_features_argyrodite_gt1e-6_train.csv",
    )
    parser.add_argument(
        "--test-output",
        type=Path,
        default=FEATURE_DIR / "ionic_8_features_argyrodite_gt1e-6_test.csv",
    )
    parser.add_argument(
        "--sample-output",
        type=Path,
        default=FEATURE_DIR / "ionic_8_features_argyrodite_gt1e-6_samples.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_argyrodite_features(
        train_input=args.train_input,
        test_input=args.test_input,
        raw_path=args.raw,
        train_output=args.train_output,
        test_output=args.test_output,
        sample_output=args.sample_output,
        subset=args.subset,
        test_size=args.test_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
