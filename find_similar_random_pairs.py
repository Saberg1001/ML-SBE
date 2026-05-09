from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
TARGET = "Ionic conductivity (S cm-1)"
OUTPUT = ROOT / "models" / "outputs" / "ionic_26_features_random" / "similar_test_train_pairs.csv"


def parse_log10(value: object) -> float:
    text = str(value).strip().replace("<", "").replace("≤", "").replace(">", "").replace("≥", "")
    if not text or text.lower() == "nan":
        return math.nan
    return math.log10(float(text))


def main() -> None:
    train = pd.read_csv(ROOT / "features" / "ionic_26_features_random_train.csv")
    test = pd.read_csv(ROOT / "features" / "ionic_26_features_random_test.csv")
    raw_train = pd.read_csv(ROOT / "rawdata" / "random_train.csv").set_index("ID")
    raw_test = pd.read_csv(ROOT / "rawdata" / "random_test.csv").set_index("ID")

    metadata = {"ID", "True Composition", "Z_by_element", TARGET}
    feature_cols = [column for column in train.columns if column not in metadata]
    x_train = train[feature_cols].apply(pd.to_numeric, errors="coerce")
    x_test = test[feature_cols].apply(pd.to_numeric, errors="coerce")
    medians = x_train.median()

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train.fillna(medians))
    x_test_scaled = scaler.transform(x_test.fillna(medians))
    distances = pairwise_distances(x_test_scaled, x_train_scaled)
    nearest_index = distances.argmin(axis=1)
    nearest_distance = distances.min(axis=1)

    rows = []
    for test_row_index in np.argsort(nearest_distance):
        if nearest_distance[test_row_index] > 0.05:
            continue
        test_id = test.loc[test_row_index, "ID"]
        train_id = train.loc[nearest_index[test_row_index], "ID"]
        test_raw = raw_test.loc[test_id]
        train_raw = raw_train.loc[train_id]
        test_log = parse_log10(test_raw[TARGET])
        train_log = parse_log10(train_raw[TARGET])
        rows.append({
            "feature_distance": nearest_distance[test_row_index],
            "test_id": test_id,
            "test_reduced_composition": test_raw["Reduced Composition"],
            "test_true_composition": test_raw["True Composition"],
            "test_family": test_raw["Family"],
            "test_conductivity": test_raw[TARGET],
            "test_log10_conductivity": test_log,
            "train_id": train_id,
            "train_reduced_composition": train_raw["Reduced Composition"],
            "train_true_composition": train_raw["True Composition"],
            "train_family": train_raw["Family"],
            "train_conductivity": train_raw[TARGET],
            "train_log10_conductivity": train_log,
            "abs_log10_diff": abs(test_log - train_log),
        })

    output = pd.DataFrame(rows)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT, index=False)
    print(f"Wrote {len(output)} similar pairs to {OUTPUT}")
    print(output.to_string(index=False))


if __name__ == "__main__":
    main()
