from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMP_DIR = PROJECT_ROOT / "comp_chick_adult"
for path in (PROJECT_ROOT, COMP_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_analysis import (
    FEATURE_COLUMNS,
    SAMPLE_LENGTH,
    SAMPLE_RATE,
    classify_phase,
    compare_features,
    parse_recording_date,
    run_analysis,
    run_recording_analysis,
)


def write_tone(path: Path, frequency: float, seconds: float = 6.0) -> None:
    samples = np.arange(round(SAMPLE_RATE * seconds), dtype=np.float32)
    audio = 0.1 * np.sin(2 * np.pi * frequency * samples / SAMPLE_RATE)
    sf.write(path, audio, SAMPLE_RATE)


class CountingPredictor:
    def __init__(self, logit: float = 3.0) -> None:
        self.logit = logit
        self.call_count = 0
        self.window_count = 0

    def predict(self, batch: np.ndarray) -> np.ndarray:
        self.call_count += 1
        self.window_count += len(batch)
        return np.full(len(batch), self.logit, dtype=np.float32)


def test_date_parsing_and_proxy_boundaries() -> None:
    assert parse_recording_date(Path("site_210531_235959.wav")) == date(2021, 5, 31)
    assert parse_recording_date(Path("site_without_date.wav")) is None
    assert (
        classify_phase(date(2021, 5, 31), date(2021, 5, 31), date(2021, 7, 23))
        == "adult_proxy"
    )
    assert classify_phase(date(2021, 6, 1), date(2021, 5, 31), date(2021, 7, 23)) is None
    assert (
        classify_phase(date(2021, 7, 23), date(2021, 5, 31), date(2021, 7, 23))
        == "chick_proxy"
    )


def test_full_analysis_features_yield_dedup_and_outputs(tmp_path: Path) -> None:
    recordings: list[tuple[Path, str]] = []
    for name, frequency, phase in [
        ("site_210501_010000.wav", 500.0, "adult_proxy"),
        ("site_210502_010000.wav", 600.0, "adult_proxy"),
        ("site_210801_010000.wav", 5_000.0, "chick_proxy"),
        ("site_210802_010000.wav", 6_000.0, "chick_proxy"),
    ]:
        path = tmp_path / name
        write_tone(path, frequency)
        recordings.append((path, phase))
    transition = tmp_path / "site_210620_010000.wav"
    write_tone(transition, 1_000.0)
    no_date = tmp_path / "undated.wav"
    write_tone(no_date, 1_000.0)

    rows: list[dict[str, object]] = []
    for recording, _ in recordings:
        rows.extend(
            [
                {
                    "recording": recording,
                    "start_seconds": 0.0,
                    "end_seconds": 3.0,
                    "probability": 0.80,
                },
                {
                    "recording": recording,
                    "start_seconds": 3.0,
                    "end_seconds": 6.0,
                    "probability": 0.60,
                },
            ]
        )
    rows.extend(
        [
            {
                "recording": transition,
                "start_seconds": 0,
                "end_seconds": 3,
                "probability": 0.99,
            },
            {
                "recording": no_date,
                "start_seconds": 0,
                "end_seconds": 3,
                "probability": 0.99,
            },
            {
                "recording": recordings[0][0],
                "start_seconds": 4,
                "end_seconds": 7,
                "probability": 0.99,
            },
            {
                "recording": tmp_path / "missing_210501_020000.wav",
                "start_seconds": 0,
                "end_seconds": 3,
                "probability": 0.99,
            },
            {
                "recording": recordings[0][0],
                "start_seconds": 0.5,
                "end_seconds": 3.0,
                "probability": 0.99,
            },
        ]
    )
    input_csv = tmp_path / "scores" / "all_windows.csv"
    input_csv.parent.mkdir()
    pd.DataFrame(rows).to_csv(input_csv, index=False)
    duplicate_csv = input_csv.parent / "second_run" / "all_windows.csv"
    duplicate_csv.parent.mkdir()
    pd.DataFrame([rows[0].copy()]).to_csv(duplicate_csv, index=False)

    output = tmp_path / "analysis"
    run_analysis(
        [input_csv.parent],
        output,
        threshold=0.80,
        representative_count=2,
        bootstrap_iterations=100,
    )

    windows = pd.read_csv(output / "analysis_windows.csv")
    assert len(windows) == 4
    assert set(windows["phase"]) == {"adult_proxy", "chick_proxy"}
    adult_centroid = windows.loc[windows["phase"] == "adult_proxy", "spectral_centroid_hz"].median()
    chick_centroid = windows.loc[windows["phase"] == "chick_proxy", "spectral_centroid_hz"].median()
    assert chick_centroid > adult_centroid

    threshold_yield = pd.read_csv(output / "threshold_yield.csv")
    for phase in ("adult_proxy", "chick_proxy"):
        assert int(
            threshold_yield.loc[
                (threshold_yield["threshold"] == 0.2) & (threshold_yield["phase"] == phase),
                "window_count",
            ].iloc[0]
        ) == 4
        assert int(
            threshold_yield.loc[
                (threshold_yield["threshold"] == 0.8) & (threshold_yield["phase"] == phase),
                "recording_count",
            ].iloc[0]
        ) == 2

    exclusions = pd.read_csv(output / "excluded_windows.csv")
    assert {
        "duplicate_recording_start",
        "incubation_transition_period",
        "unparseable_recording_date",
        "not_complete_three_second_window",
        "source_file_missing",
        "source_audio_shorter_than_window",
        "below_analysis_threshold",
    }.issubset(set(exclusions["exclusion_reason"]))

    recording_summary = pd.read_csv(output / "recording_summary.csv")
    comparison = pd.read_csv(output / "feature_comparison.csv")
    assert len(recording_summary) == 4
    assert comparison["p_value"].notna().all()
    assert (comparison["adult_recording_count"] == 2).all()
    assert (comparison["chick_recording_count"] == 2).all()

    wavs = list((output / "representative_audio").rglob("*.wav"))
    assert len(wavs) == 4
    for wav in wavs:
        audio, rate = sf.read(wav)
        assert rate == SAMPLE_RATE
        assert audio.ndim == 1
        assert len(audio) == SAMPLE_LENGTH
    expected_figures = {
        "average_log_mel_spectrograms.png",
        "difference_log_mel_spectrogram.png",
        "representative_spectrograms.png",
        "feature_distributions.png",
        "probability_distributions.png",
        "date_activity.png",
    }
    assert expected_figures == {path.name for path in (output / "figures").glob("*.png")}
    assert (output / "analysis_report.md").is_file()


def test_inference_is_suppressed_when_either_phase_has_one_recording() -> None:
    rows = []
    for phase, offset in (("adult_proxy", 0.0), ("chick_proxy", 1.0)):
        row: dict[str, object] = {
            "recording": f"{phase}.wav",
            "recording_date": "2021-01-01",
            "phase": phase,
            "recording_duration_seconds": 3600.0,
            "eligible_window_count": 1,
            "high_confidence_window_count": 1,
            "high_confidence_hits_per_hour": 1.0 + offset,
            "median_probability": 0.9,
        }
        row.update({feature: 1.0 + offset for feature in FEATURE_COLUMNS})
        rows.append(row)
    comparison, enough = compare_features(pd.DataFrame(rows), bootstrap_iterations=20, seed=1)
    assert not enough
    assert comparison["p_value"].isna().all()
    assert comparison["p_value_bh_fdr"].isna().all()


def test_independent_recording_scan_cache_and_rescan(tmp_path: Path) -> None:
    recordings_dir = tmp_path / "recordings"
    recordings_dir.mkdir()
    selected = [
        ("site_210501_010000.wav", 500.0),
        ("site_210502_010000.wav", 600.0),
        ("site_210801_010000.wav", 5_000.0),
        ("site_210802_010000.wav", 6_000.0),
    ]
    for name, frequency in selected:
        write_tone(recordings_dir / name, frequency, seconds=3.0)
    write_tone(recordings_dir / "site_210620_010000.wav", 1_000.0, seconds=3.0)
    write_tone(recordings_dir / "undated.wav", 1_000.0, seconds=3.0)
    write_tone(recordings_dir / "short_210501_030000.wav", 1_000.0, seconds=2.0)

    model = tmp_path / "model.tflite"
    model.write_bytes(b"fake-model-v1")
    model.with_name("model_Labels.txt").write_text(
        "not_storm_petrel\nstorm_petrel\n",
        encoding="utf-8",
    )
    output = tmp_path / "analysis"
    predictor = CountingPredictor()

    _, reused = run_recording_analysis(
        recordings_dir,
        output,
        model_path=model,
        threshold=0.80,
        representative_count=1,
        bootstrap_iterations=20,
        predictor=predictor,
    )
    assert not reused
    assert predictor.window_count == 4
    assert len(pd.read_csv(output / "scan" / "all_windows.csv")) == 4
    excluded_recordings = pd.read_csv(output / "scan" / "excluded_recordings.csv")
    assert set(excluded_recordings["exclusion_reason"]) == {
        "incubation_transition_period",
        "source_audio_shorter_than_window",
        "unparseable_recording_date",
    }
    first_call_count = predictor.call_count

    # A changed analysis threshold refreshes derived outputs without model inference.
    _, reused = run_recording_analysis(
        recordings_dir,
        output,
        model_path=model,
        threshold=0.90,
        representative_count=1,
        bootstrap_iterations=20,
        predictor=predictor,
    )
    assert reused
    assert predictor.call_count == first_call_count
    assert len(pd.read_csv(output / "analysis_windows.csv")) == 4

    # Model metadata changes invalidate the expensive score cache.
    model.write_bytes(b"fake-model-v2-with-different-size")
    try:
        run_recording_analysis(
            recordings_dir,
            output,
            model_path=model,
            threshold=0.90,
            representative_count=1,
            bootstrap_iterations=20,
            predictor=predictor,
        )
    except ValueError as exc:
        assert "--rescan" in str(exc)
    else:
        raise AssertionError("Expected changed model metadata to invalidate the scan cache")

    _, reused = run_recording_analysis(
        recordings_dir,
        output,
        model_path=model,
        threshold=0.90,
        representative_count=1,
        bootstrap_iterations=20,
        rescan=True,
        predictor=predictor,
    )
    assert not reused
    assert predictor.call_count > first_call_count
