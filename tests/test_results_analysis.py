from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from results_analysis.summarize_nested_cv import clustered_bootstrap_ci, pooled_metrics


def test_clustered_bootstrap_is_deterministic_and_reports_pooled_estimates() -> None:
    rows = []
    probabilities = {
        "source_a.wav": (0.95, 0.10),
        "source_b.wav": (0.90, 0.20),
        "source_c.wav": (0.80, 0.30),
        "source_d.wav": (0.40, 0.60),
    }
    for source_file, (positive_probability, negative_probability) in probabilities.items():
        rows.extend(
            [
                {"source_file": source_file, "true_label": 1, "probability": positive_probability},
                {"source_file": source_file, "true_label": 0, "probability": negative_probability},
            ]
        )
    oof = pd.DataFrame(rows)

    first = clustered_bootstrap_ci(oof, iterations=100, seed=2021)
    second = clustered_bootstrap_ci(oof, iterations=100, seed=2021)
    pd.testing.assert_frame_equal(first, second)

    assert first["metric"].tolist() == ["auroc", "auprc", "precision", "recall", "f1"]
    assert set(first["bootstrap_unit"]) == {"source_file"}
    assert set(first["iterations"]) == {100}
    assert set(first["seed"]) == {2021}
    expected = pooled_metrics(oof)
    assert first.set_index("metric")["estimate"].to_dict() == expected
