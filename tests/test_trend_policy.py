"""Tests for the fixed trend-label and direction-symmetry policies."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from main.trend.features import (
    B_BASELINE_FEATURES,
    MODEL_FEATURE_COLUMNS,
    build_prediction_features,
    classify_trend_delta,
)
from main.trend.predict import _grouped_raw_rows
from main.trend.train_classifier import (
    ClassifierConfig,
    _grouped_folds,
    _reverse_frame,
    train_classifiers,
)


class TrendLabelPolicyTests(unittest.TestCase):
    def test_repeated_group_names_remain_separate_segments(self) -> None:
        with TemporaryDirectory() as directory:
            raw_path = Path(directory) / "experimental.csv"
            raw_path.write_text(
                "ID\tformula\tconductivity\n"
                "group8\n"
                "exp_001\tLi3YCl6\t1e-3\n"
                "exp_002\tLi2ZrCl6\t2e-3\n"
                "group8\n"
                "exp_003\tLi3InCl6\t3e-3\n",
                encoding="utf-8",
            )
            rows = _grouped_raw_rows(raw_path)

        self.assertEqual([row["group_id"] for row in rows], ["group8"] * 3)
        self.assertEqual(
            [row["group_segment"] for row in rows],
            ["1", "1", "2"],
        )

    def test_explicit_b_baseline_features_are_included(self) -> None:
        self.assertEqual(len(B_BASELINE_FEATURES), 12)
        self.assertEqual(len(MODEL_FEATURE_COLUMNS), 54)
        self.assertTrue(set(B_BASELINE_FEATURES) <= set(MODEL_FEATURE_COLUMNS))

    def test_absolute_threshold_and_boundaries(self) -> None:
        delta = np.array([-1.1e-4, -1e-4, 0.0, 1e-4, 1.1e-4])
        labels = classify_trend_delta(delta)
        self.assertEqual(
            labels.tolist(),
            ["decrease", "unchanged", "unchanged", "unchanged", "increase"],
        )

    def test_alternate_threshold_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "fixed absolute threshold"):
            classify_trend_delta([0.0], threshold_s_cm=1e-3)

    def test_training_cannot_disable_swap_augmentation(self) -> None:
        config = ClassifierConfig(models=(), swap_augmentation=False)
        with self.assertRaisesRegex(ValueError, "requires swap augmentation"):
            train_classifiers(config)

    def test_swap_matches_features_built_in_reverse_order(self) -> None:
        forward = build_prediction_features(
            "Li3YCl6", "Li2.9Y0.9Zr0.1Cl6", "halides"
        )
        forward["trend_label"] = "increase"
        reversed_frame = _reverse_frame(forward)
        expected = build_prediction_features(
            "Li2.9Y0.9Zr0.1Cl6", "Li3YCl6", "halides"
        )
        np.testing.assert_allclose(
            reversed_frame[MODEL_FEATURE_COLUMNS].to_numpy(dtype=float),
            expected[MODEL_FEATURE_COLUMNS].to_numpy(dtype=float),
        )
        self.assertEqual(reversed_frame.loc[0, "trend_label"], "decrease")

    def test_grouped_cv_keeps_dois_whole_and_covers_each_row_once(self) -> None:
        rows = []
        for group_index in range(15):
            for label in ("decrease", "unchanged", "increase"):
                rows.append({"doi": f"doi-{group_index}", "trend_label": label})
        frame = pd.DataFrame(rows)
        folds = _grouped_folds(frame, ClassifierConfig(cv_splits=5, seed=42))
        validation_counts = np.zeros(len(frame), dtype=int)
        for fit_index, valid_index in folds:
            fit_dois = set(frame.iloc[fit_index]["doi"])
            valid_dois = set(frame.iloc[valid_index]["doi"])
            self.assertFalse(fit_dois & valid_dois)
            validation_counts[valid_index] += 1
        np.testing.assert_array_equal(validation_counts, np.ones(len(frame), dtype=int))


if __name__ == "__main__":
    unittest.main()
