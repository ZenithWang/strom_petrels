from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LONG_ANALYSIS_DIR = PROJECT_ROOT / "long_recording_analysis"
for path in (PROJECT_ROOT, LONG_ANALYSIS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from extract_candidates import extract_candidates
from scan_long_recordings import SAMPLE_LENGTH, SAMPLE_RATE, iter_audio_windows, scan_recordings, sigmoid


class FakePredictor:
    def predict(self, batch: np.ndarray) -> np.ndarray:
        assert batch.shape[1] == SAMPLE_LENGTH
        return np.linspace(-2.0, 2.0, len(batch), dtype=np.float32)


def write_tone(path: Path, sample_rate: int, seconds: float, channels: int = 1) -> None:
    samples = np.arange(round(sample_rate * seconds), dtype=np.float32)
    audio = 0.1 * np.sin(2 * np.pi * 440 * samples / sample_rate)
    if channels == 2:
        audio = np.column_stack((audio, audio * 0.5))
    sf.write(path, audio, sample_rate)


def test_windowing_resamples_mixes_and_ignores_short_tail(tmp_path: Path) -> None:
    source = tmp_path / "stereo_32k.flac"
    write_tone(source, 32_000, 7.0, channels=2)
    windows = list(iter_audio_windows(source))
    assert [item.start_seconds for item in windows] == [0.0, 3.0]
    assert [item.end_seconds for item in windows] == [3.0, 6.0]
    assert all(item.audio.shape == (SAMPLE_LENGTH,) for item in windows)


def test_scan_probabilities_and_extract_without_rescanning(tmp_path: Path) -> None:
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    recording = input_dir / "short.wav"
    write_tone(recording, SAMPLE_RATE, 6.0)
    scores = tmp_path / "run" / "all_windows.csv"
    count = scan_recordings(input_dir, scores, tmp_path / "unused.tflite", predictor=FakePredictor())
    assert count == 2
    with scores.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert all(0.0 <= float(row["probability"]) <= 1.0 for row in rows)
    assert float(rows[0]["probability"]) == round(sigmoid(-2.0), 8)

    detections = scores.parent / "detections.csv"
    candidates = scores.parent / "candidates"
    result = extract_candidates(scores, detections, candidates, threshold=0.20)
    assert result == 1
    with detections.open(newline="", encoding="utf-8") as handle:
        detection = next(csv.DictReader(handle))
    assert float(detection["start_seconds"]) == 3.0
    assert float(detection["end_seconds"]) == 6.0
    assert int(detection["window_count"]) == 1
    candidate_audio, candidate_rate = sf.read(detection["candidate_path"])
    assert candidate_rate == SAMPLE_RATE
    assert candidate_audio.ndim == 1
    assert len(candidate_audio) == SAMPLE_LENGTH

    # A threshold-only rerun reads all_windows.csv and replaces extraction outputs.
    second = extract_candidates(scores, detections, candidates, threshold=0.90, force=True)
    assert second == 0
    assert scores.is_file()
