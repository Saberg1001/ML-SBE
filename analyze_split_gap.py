from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
TARGET = "Ionic conductivity (S cm-1)"
META_COLUMNS = {"ID", "True Composition", "Z_by_element", TARGET}
OUTPUT = ROOT / "models" / "outputs" / "split_gap_analysis"


SPLITS = [
    ("fixed", "ionic_26_features", "features/ionic_26_features_train.csv", "features/ionic_26_features_test.csv"),
    ("random", "ionic_26_features_random", "features/ionic_26_features_random_train.csv", "features/ionic_26_features_random_test.csv"),
]


def nearest_distances(train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    feature_cols = [column for column in train.columns if column not in META_COLUMNS]
    x_train = train[feature_cols].apply(pd.to_numeric, errors="coerce")
    x_test = test[feature_cols].apply(pd.to_numeric, errors="coerce")
    medians = x_train.median()
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train.fillna(medians))
    x_test_scaled = scaler.transform(x_test.fillna(medians))
    distances = pairwise_distances(x_test_scaled, x_train_scaled)
    return distances.min(axis=1), distances.argmin(axis=1)


def coverage_row(name: str, raw_train: pd.DataFrame, raw_test: pd.DataFrame) -> dict[str, object]:
    row: dict[str, object] = {"split": name, "test_n": len(raw_test)}
    for column in ["Reduced Composition", "True Composition", "Family", "DOI", "Space group #"]:
        train_values = set(raw_train[column].dropna().astype(str).str.strip())
        test_values = raw_test[column].dropna().astype(str).str.strip()
        row[f"{column}_covered"] = int(test_values.isin(train_values).sum())
    return row


def raw_rows_for_ids(ids: pd.Series) -> pd.DataFrame:
    raw_all = pd.read_csv(ROOT / "rawdata" / "all.csv").set_index("ID")
    return raw_all.loc[ids].reset_index()


def summarize_split(name: str, train: pd.DataFrame, test: pd.DataFrame, raw_train: pd.DataFrame, raw_test: pd.DataFrame) -> dict[str, object]:
    distances, _ = nearest_distances(train, test)
    row = coverage_row(name, raw_train, raw_test)
    row.update({
        "distance_mean": distances.mean(),
        "distance_median": np.median(distances),
        "distance_p75": np.quantile(distances, 0.75),
        "distance_p90": np.quantile(distances, 0.90),
        "distance_max": distances.max(),
        "distance_le_0.05": int((distances <= 0.05).sum()),
        "distance_le_0.25": int((distances <= 0.25).sum()),
        "distance_le_1.0": int((distances <= 1.0).sum()),
    })
    return row


def fixed_error_neighbors() -> pd.DataFrame:
    train = pd.read_csv(ROOT / "features" / "ionic_26_features_train.csv")
    test = pd.read_csv(ROOT / "features" / "ionic_26_features_test.csv")
    raw_train = raw_rows_for_ids(train["ID"]).set_index("ID")
    raw_test = raw_rows_for_ids(test["ID"]).set_index("ID")
    predictions = pd.read_csv(ROOT / "models" / "outputs" / "ionic_26_features" / "lightgbm" / "test_predictions.csv")
    distances, nearest_index = nearest_distances(train, test)
    test_position = {id_: index for index, id_ in enumerate(test["ID"])}

    rows = []
    for _, prediction in predictions.assign(abs_error=lambda frame: frame["residual"].abs()).sort_values("abs_error", ascending=False).head(20).iterrows():
        test_id = prediction["ID"]
        index = test_position[test_id]
        train_id = train.loc[nearest_index[index], "ID"]
        test_raw = raw_test.loc[test_id]
        train_raw = raw_train.loc[train_id]
        rows.append({
            "test_id": test_id,
            "test_reduced_composition": test_raw["Reduced Composition"],
            "test_family": test_raw["Family"],
            "test_conductivity": test_raw[TARGET],
            "y_true": prediction["y_true"],
            "y_pred": prediction["y_pred"],
            "abs_error": prediction["abs_error"],
            "nearest_distance": distances[index],
            "nearest_train_id": train_id,
            "nearest_train_composition": train_raw["Reduced Composition"],
            "nearest_train_family": train_raw["Family"],
            "nearest_train_conductivity": train_raw[TARGET],
        })
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for name, _, train_path, test_path in SPLITS:
        train = pd.read_csv(ROOT / train_path)
        test = pd.read_csv(ROOT / test_path)
        raw_train = raw_rows_for_ids(train["ID"])
        raw_test = raw_rows_for_ids(test["ID"])
        summary_rows.append(summarize_split(name, train, test, raw_train, raw_test))

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUTPUT / "split_similarity_summary.csv", index=False)
    errors = fixed_error_neighbors()
    errors.to_csv(OUTPUT / "fixed_lightgbm_top_error_neighbors.csv", index=False)
    print(summary.to_string(index=False))
    print()
    print(errors.to_string(index=False))


if __name__ == "__main__":
    main()
