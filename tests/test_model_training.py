from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model_training.run_nested_cv import SAMPLE_LENGTH, SAMPLE_RATE, load_audio, metric_from_evaluation


def test_missing_inner_evaluation_fails_loudly(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Missing BirdNET evaluation file"):
        metric_from_evaluation(tmp_path / "missing_evaluation.csv")


def test_empty_or_metricless_inner_evaluation_fails_loudly(tmp_path: Path) -> None:
    empty = tmp_path / "empty.csv"
    empty.write_text("Class,F1 Score (opt)\n", encoding="utf-8")
    with pytest.raises(ValueError, match="is empty"):
        metric_from_evaluation(empty)

    metricless = tmp_path / "metricless.csv"
    pd.DataFrame([{"Class": "storm_petrel", "Precision": 1.0}]).to_csv(metricless, index=False)
    with pytest.raises(ValueError, match="no supported F1 metric"):
        metric_from_evaluation(metricless)

    wrong_class = tmp_path / "wrong_class.csv"
    pd.DataFrame([{"Class": "not_storm_petrel", "F1 Score (opt)": 1.0}]).to_csv(wrong_class, index=False)
    with pytest.raises(ValueError, match="no 'storm_petrel' row"):
        metric_from_evaluation(wrong_class)


def test_cv_evaluation_rejects_short_clip_instead_of_padding(tmp_path: Path) -> None:
    path = tmp_path / "short.wav"
    sf.write(path, np.zeros(SAMPLE_LENGTH - 1, dtype=np.float32), SAMPLE_RATE)
    with pytest.raises(ValueError, match="shorter.*padding is disabled"):
        load_audio(path)
