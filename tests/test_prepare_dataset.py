from __future__ import annotations

import io
import sys
import unittest
from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


DATA_PROCESSING_DIR = Path(__file__).resolve().parents[1] / "data_processing"
if str(DATA_PROCESSING_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_PROCESSING_DIR))

import prepare_dataset


def candidate(
    start: float,
    score: float = 1.0,
    *,
    source: str = "recording.wav",
    db_label: str = "positive",
    target_label: str = "1",
    negative_type: str = "",
) -> prepare_dataset.Candidate:
    return prepare_dataset.Candidate(
        db_label=db_label,
        species="storm_petrel" if target_label == "1" else db_label,
        score=score,
        location=1,
        timestamp="2021-01-01",
        recording_file=source,
        duration_seconds=60.0,
        start_seconds=start,
        end_seconds=start + 3.0,
        label="storm_petrel" if target_label == "1" else "not_storm_petrel",
        target_label=target_label,
        negative_type=negative_type,
    )


class SampleSelectionTests(unittest.TestCase):
    def test_dataset_targets_are_balanced(self) -> None:
        class_counts = Counter()
        negative_counts = Counter()
        for target in prepare_dataset.LABEL_TARGETS:
            class_counts[target["target_label"]] += target["count"]
            if target["target_label"] == "0":
                negative_counts[target["negative_type"]] += target["count"]
        self.assertEqual(class_counts, {"1": 1000, "0": 1000})
        self.assertEqual(negative_counts, {"environment_noise": 500, "other_bird": 500})

    def test_selection_is_source_balanced_and_windows_do_not_overlap(self) -> None:
        targets = (
            {
                "db_label": "positive",
                "count": 2,
                "label": "storm_petrel",
                "target_label": "1",
                "negative_type": "",
            },
            {
                "db_label": "negative_a",
                "count": 1,
                "label": "not_storm_petrel",
                "target_label": "0",
                "negative_type": "other_bird",
            },
            {
                "db_label": "negative_b",
                "count": 1,
                "label": "not_storm_petrel",
                "target_label": "0",
                "negative_type": "environment_noise",
            },
        )
        candidates_by_label = {
            "positive": [candidate(0, 1.0), candidate(1, 0.9), candidate(3, 0.8)],
            "negative_a": [
                candidate(6, 0.9, db_label="negative_a", target_label="0", negative_type="other_bird")
            ],
            "negative_b": [
                candidate(
                    6,
                    0.99,
                    db_label="negative_b",
                    target_label="0",
                    negative_type="environment_noise",
                ),
                candidate(
                    9,
                    0.8,
                    db_label="negative_b",
                    target_label="0",
                    negative_type="environment_noise",
                ),
            ],
        }

        def read_candidates(db_label: str, *_args: str) -> list[prepare_dataset.Candidate]:
            return candidates_by_label[db_label]

        with (
            patch.object(prepare_dataset, "LABEL_TARGETS", targets),
            patch.object(prepare_dataset, "read_candidates", side_effect=read_candidates),
        ):
            selected = prepare_dataset.select_samples()
        self.assertEqual(Counter(item.target_label for item in selected), {"1": 2, "0": 2})
        self.assertEqual(sorted(item.start_seconds for item in selected), [0, 3, 6, 9])

    def test_selection_uses_highest_positive_score_available_per_source(self) -> None:
        targets = (
            {
                "db_label": "positive",
                "count": 2,
                "label": "storm_petrel",
                "target_label": "1",
                "negative_type": "",
            },
            {
                "db_label": "negative",
                "count": 2,
                "label": "not_storm_petrel",
                "target_label": "0",
                "negative_type": "other_bird",
            },
        )
        candidates_by_label = {
            "positive": [
                candidate(0, 0.99, source="a.wav"),
                candidate(3, 0.60, source="a.wav"),
                candidate(0, 0.95, source="b.wav"),
                candidate(3, 0.50, source="b.wav"),
            ],
            "negative": [
                candidate(
                    9,
                    0.9,
                    source="a.wav",
                    db_label="negative",
                    target_label="0",
                    negative_type="other_bird",
                ),
                candidate(
                    9,
                    0.8,
                    source="b.wav",
                    db_label="negative",
                    target_label="0",
                    negative_type="other_bird",
                ),
            ],
        }

        def read_candidates(db_label: str, *_args: str) -> list[prepare_dataset.Candidate]:
            return candidates_by_label[db_label]

        with (
            patch.object(prepare_dataset, "LABEL_TARGETS", targets),
            patch.object(prepare_dataset, "read_candidates", side_effect=read_candidates),
        ):
            selected = prepare_dataset.select_samples()
        positive_scores = sorted(item.score for item in selected if item.target_label == "1")
        self.assertEqual(positive_scores, [0.95, 0.99])

    def test_insufficient_non_overlapping_candidates_fails(self) -> None:
        targets = (
            {
                "db_label": "positive",
                "count": 3,
                "label": "storm_petrel",
                "target_label": "1",
                "negative_type": "",
            },
        )
        candidates = [candidate(0, 1.0), candidate(1, 0.9), candidate(3, 0.8)]
        with (
            patch.object(prepare_dataset, "LABEL_TARGETS", targets),
            patch.object(prepare_dataset, "read_candidates", return_value=candidates),
            self.assertRaisesRegex(prepare_dataset.SampleSelectionError, "infeasible"),
        ):
            prepare_dataset.select_samples()

    def test_selection_rejects_feasible_solution_not_proven_optimal(self) -> None:
        targets = (
            {
                "db_label": "positive",
                "count": 1,
                "label": "storm_petrel",
                "target_label": "1",
                "negative_type": "",
            },
            {
                "db_label": "negative",
                "count": 1,
                "label": "not_storm_petrel",
                "target_label": "0",
                "negative_type": "other_bird",
            },
        )
        positive_pool = {"recording.wav": [candidate(0)]}
        negative_pool = {
            "recording.wav": [
                candidate(3, db_label="negative", target_label="0", negative_type="other_bird")
            ]
        }
        timed_out = SimpleNamespace(
            success=True,
            status=1,
            x=[1.0, 1.0],
            fun=-1.0,
            message="Time limit reached with a feasible incumbent",
        )
        with (
            patch.object(prepare_dataset, "LABEL_TARGETS", targets),
            patch.object(prepare_dataset, "milp", return_value=timed_out),
            self.assertRaisesRegex(prepare_dataset.SampleSelectionError, "proven optimum.*status=1"),
        ):
            prepare_dataset.optimize_balanced_selection(positive_pool, negative_pool)


class FoldAssignmentTests(unittest.TestCase):
    @staticmethod
    def balanced_rows() -> list[dict[str, str]]:
        rows = []
        for group_index in range(30):
            source = f"source_{group_index:02d}.wav"
            rows.append({"source_file": source, "target_label": "1", "negative_type": ""})
            rows.append(
                {
                    "source_file": source,
                    "target_label": "0",
                    "negative_type": "environment_noise" if group_index % 2 else "other_bird",
                }
            )
        return rows

    def test_nested_plan_is_balanced_grouped_and_deterministic(self) -> None:
        rows = self.balanced_rows()
        with redirect_stdout(io.StringIO()):
            first = prepare_dataset.build_fold_plan(rows)
            second = prepare_dataset.build_fold_plan(rows)
        self.assertEqual(first, second)

        fold_rows = prepare_dataset.build_fold_rows(rows, first)
        report = prepare_dataset.build_balance_report_rows(fold_rows)
        self.assertEqual(len(report), 40)
        for item in report:
            self.assertEqual(float(item["positive_ratio"]), 0.50)

        for outer_fold in range(1, prepare_dataset.OUTER_FOLDS + 1):
            train_sources = {
                row["source_file"]
                for row in rows
                if first.outer_assignments[row["source_file"]] != outer_fold
            }
            test_sources = {
                row["source_file"]
                for row in rows
                if first.outer_assignments[row["source_file"]] == outer_fold
            }
            self.assertFalse(train_sources & test_sources)

    def test_unbalanced_source_group_fails(self) -> None:
        rows = [
            {"source_file": "positive_block.wav", "target_label": "1", "negative_type": ""}
            for _ in range(50)
        ]
        for group_index in range(10):
            rows.extend(
                {
                    "source_file": f"negative_{group_index:02d}.wav",
                    "target_label": "0",
                    "negative_type": "environment_noise" if group_index % 2 else "other_bird",
                }
                for _ in range(5)
            )
        with self.assertRaisesRegex(prepare_dataset.FoldAssignmentError, "requires every source group"):
            prepare_dataset.assign_groups_to_folds(rows, 5, "outer")


if __name__ == "__main__":
    unittest.main()
