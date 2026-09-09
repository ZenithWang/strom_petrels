from __future__ import annotations

import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
CHICK_DIR = SCRIPTS_DIR / "chick_screening"
if str(CHICK_DIR) not in sys.path:
    sys.path.insert(0, str(CHICK_DIR))
DATA_PROCESSING_DIR = SCRIPTS_DIR / "data_processing"
if str(DATA_PROCESSING_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_PROCESSING_DIR))

from analyze_discrepancies import (
    QUADRANT_LABELS,
    build_quadrant_summary,
    classify_quadrant,
    compare_quadrant_features,
)
from dual_model_scan import (
    DUAL_CSV_FIELDS,
    sigmoid,
)
from prepare_dataset import Candidate, _overlap_pairs


class QuadrantClassificationTests(unittest.TestCase):
    def test_all_four_quadrants(self) -> None:
        self.assertEqual(classify_quadrant(0.3, 0.9, 0.8, 0.8), "chick_candidate")
        self.assertEqual(classify_quadrant(0.9, 0.9, 0.8, 0.8), "general_storm_petrel")
        self.assertEqual(classify_quadrant(0.9, 0.3, 0.8, 0.8), "adult_biased")
        self.assertEqual(classify_quadrant(0.3, 0.3, 0.8, 0.8), "background")

    def test_boundary_values(self) -> None:
        # Exactly at threshold counts as high
        self.assertEqual(classify_quadrant(0.80, 0.80, 0.80, 0.80), "general_storm_petrel")
        self.assertEqual(classify_quadrant(0.79, 0.80, 0.80, 0.80), "chick_candidate")
        self.assertEqual(classify_quadrant(0.80, 0.79, 0.80, 0.80), "adult_biased")

    def test_different_thresholds_per_model(self) -> None:
        self.assertEqual(classify_quadrant(0.70, 0.70, 0.60, 0.90), "adult_biased")
        self.assertEqual(classify_quadrant(0.70, 0.70, 0.90, 0.60), "chick_candidate")

    def test_all_labels_present(self) -> None:
        for quad in ("chick_candidate", "general_storm_petrel", "adult_biased", "background"):
            self.assertIn(quad, QUADRANT_LABELS)


class SigmoidTests(unittest.TestCase):
    def test_sigmoid_symmetry(self) -> None:
        self.assertAlmostEqual(sigmoid(0.0), 0.5, places=6)
        self.assertAlmostEqual(sigmoid(2.0) + sigmoid(-2.0), 1.0, places=6)

    def test_sigmoid_monotonic(self) -> None:
        values = [sigmoid(x) for x in np.linspace(-5, 5, 100)]
        for i in range(len(values) - 1):
            self.assertLess(values[i], values[i + 1])

    def test_sigmoid_range(self) -> None:
        for x in [-10, -1, 0, 1, 10]:
            p = sigmoid(x)
            self.assertGreater(p, 0)
            self.assertLess(p, 1)


class DualCSVFieldsTests(unittest.TestCase):
    def test_all_required_columns(self) -> None:
        required = {
            "recording", "start_seconds", "end_seconds",
            "logit_adult", "prob_adult", "logit_chick", "prob_chick",
            "delta_prob", "delta_logit",
        }
        self.assertTrue(required.issubset(set(DUAL_CSV_FIELDS)))


class QuadrantSummaryTests(unittest.TestCase):
    def _sample_scores(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"recording": "a.flac", "start_seconds": 0.0, "prob_adult": 0.9, "prob_chick": 0.9, "delta_prob": 0.0, "quadrant": "general_storm_petrel"},
            {"recording": "a.flac", "start_seconds": 3.0, "prob_adult": 0.2, "prob_chick": 0.9, "delta_prob": 0.7, "quadrant": "chick_candidate"},
            {"recording": "b.flac", "start_seconds": 0.0, "prob_adult": 0.1, "prob_chick": 0.9, "delta_prob": 0.8, "quadrant": "chick_candidate"},
            {"recording": "b.flac", "start_seconds": 3.0, "prob_adult": 0.1, "prob_chick": 0.1, "delta_prob": 0.0, "quadrant": "background"},
            {"recording": "b.flac", "start_seconds": 6.0, "prob_adult": 0.9, "prob_chick": 0.2, "delta_prob": -0.7, "quadrant": "adult_biased"},
        ])

    def test_quadrant_counts(self) -> None:
        summary = build_quadrant_summary(self._sample_scores())
        counts = dict(zip(summary["quadrant"], summary["window_count"]))
        self.assertEqual(counts["chick_candidate"], 2)
        self.assertEqual(counts["general_storm_petrel"], 1)
        self.assertEqual(counts["adult_biased"], 1)
        self.assertEqual(counts["background"], 1)

    def test_quadrant_recording_counts(self) -> None:
        summary = build_quadrant_summary(self._sample_scores())
        counts = dict(zip(summary["quadrant"], summary["recording_count"]))
        self.assertEqual(counts["chick_candidate"], 2)  # appears in both recordings


class FeatureComparisonTests(unittest.TestCase):
    def _sample_features(self) -> pd.DataFrame:
        rng = np.random.default_rng(42)
        rows = []
        for quad, n, offset in [("chick_candidate", 10, 500), ("general_storm_petrel", 15, 200)]:
            for i in range(n):
                rows.append({
                    "recording": f"rec{i%3}.flac",
                    "start_seconds": i * 3.0,
                    "quadrant": quad,
                    "prob_adult": 0.5,
                    "prob_chick": 0.5,
                    "delta_prob": 0.0,
                    "rms": float(rng.normal(0.05, 0.01)),
                    "dbfs": float(rng.normal(-20, 2)),
                    "peak_frequency_hz": float(rng.normal(offset, 100)),
                    "spectral_centroid_hz": float(rng.normal(offset, 200)),
                    "spectral_bandwidth_hz": float(rng.normal(1000, 200)),
                    "rolloff_85_hz": float(rng.normal(2000, 300)),
                    "spectral_flatness": float(rng.uniform(0, 0.1)),
                    "spectral_entropy": float(rng.uniform(0.5, 0.9)),
                    "zero_crossing_rate": float(rng.uniform(0.01, 0.1)),
                })
        return pd.DataFrame(rows)

    def test_comparison_has_all_features(self) -> None:
        comparison = compare_quadrant_features(self._sample_features(), 100, 2021)
        expected_features = set(
            sys.modules["analyze_discrepancies"].FEATURE_COLUMNS
        )
        self.assertEqual(set(comparison["feature"]), expected_features)

    def test_comparison_counts_match_input(self) -> None:
        features = self._sample_features()
        comparison = compare_quadrant_features(features, 100, 2021)
        row = comparison[comparison["feature"] == "peak_frequency_hz"].iloc[0]
        self.assertEqual(int(row["general_storm_petrel_count"]), 15)
        self.assertEqual(int(row["chick_candidate_count"]), 10)

    def test_p_values_present_with_enough_data(self) -> None:
        comparison = compare_quadrant_features(self._sample_features(), 100, 2021)
        self.assertFalse(comparison["p_value"].isna().all())


class DateFilterTests(unittest.TestCase):
    """Test that read_candidates supports the new date parameters."""

    def test_read_candidates_accepts_date_params(self) -> None:
        import inspect
        sig = inspect.signature(
            sys.modules["prepare_dataset"].read_candidates
        )
        self.assertIn("date_end", sig.parameters)
        self.assertIn("date_start", sig.parameters)

    def test_date_params_are_optional(self) -> None:
        import inspect
        sig = inspect.signature(
            sys.modules["prepare_dataset"].read_candidates
        )
        for name in ("date_end", "date_start"):
            self.assertIs(sig.parameters[name].default, None)


if __name__ == "__main__":
    unittest.main()
