"""Tests for deterministic, class-balanced DOI splitting."""

from __future__ import annotations

import unittest

import pandas as pd

from main.trend.split import _choose_dois, _target_label_counts


class TrendSplitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.counts = pd.DataFrame(
            {
                "decrease": [8, 2, 0, 0, 1, 1],
                "unchanged": [0, 0, 8, 2, 1, 1],
                "increase": [0, 0, 0, 0, 8, 2],
            },
            index=[f"doi-{index}" for index in range(6)],
        )

    def test_target_counts_sum_to_target_rows(self) -> None:
        totals = self.counts.sum(axis=0)
        target = _target_label_counts(totals, 6)
        self.assertEqual(sum(target.values()), 6)

    def test_selection_is_deterministic_and_keeps_dois_whole(self) -> None:
        target = _target_label_counts(self.counts.sum(axis=0), 6)
        first = _choose_dois(self.counts, 6, target)
        second = _choose_dois(self.counts, 6, target)
        self.assertEqual(first, second)
        self.assertTrue(first)
        self.assertLess(len(first), len(self.counts))
        selected = self.counts.loc[list(first)].sum(axis=0)
        self.assertTrue((selected > 0).all())
        self.assertTrue(((self.counts.sum(axis=0) - selected) > 0).all())


if __name__ == "__main__":
    unittest.main()
