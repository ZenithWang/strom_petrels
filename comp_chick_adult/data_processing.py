from __future__ import annotations

# Data preparation and acoustic feature extraction for the comparison workflow.

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import date, datetime
from math import gcd
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from project_config import NUMBA_CACHE_DIR
from long_recording_analysis.scan_long_recordings import (
    BatchPredictor,
    find_recordings,
    labels_path_for_model,
    scan_recording_paths,
)

os.environ.setdefault("NUMBA_CACHE_DIR", str(NUMBA_CACHE_DIR))

import librosa

SAMPLE_RATE = 48_000
WINDOW_SECONDS = 3.0
SAMPLE_LENGTH = int(SAMPLE_RATE * WINDOW_SECONDS)
DEFAULT_THRESHOLD = 0.80
DEFAULT_ADULT_PROXY_END = date(2021, 5, 31)
DEFAULT_CHICK_PROXY_START = date(2021, 7, 23)
PHASES = ("adult_proxy", "chick_proxy")
FEATURE_COLUMNS = [
    "rms",
    "dbfs",
    "peak_frequency_hz",
    "spectral_centroid_hz",
    "spectral_bandwidth_hz",
    "rolloff_85_hz",
    "spectral_flatness",
    "spectral_entropy",
    "zero_crossing_rate",
]
ELIGIBLE_COLUMNS = [
    "input_csv",
    "recording",
    "recording_date",
    "phase",
    "start_seconds",
    "end_seconds",
    "probability",
    "recording_duration_seconds",
]
ANALYSIS_COLUMNS = ELIGIBLE_COLUMNS + FEATURE_COLUMNS
REQUIRED_COLUMNS = {"recording", "start_seconds", "end_seconds", "probability"}
DATE_PATTERN = re.compile(r"(?<!\d)(\d{6})_(\d{6})(?!\d)")
SCAN_CONFIG_VERSION = 1


def parse_iso_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Expected YYYY-MM-DD, got {value!r}") from exc


def validate_name(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value):
        raise ValueError("analysis_name may contain only letters, digits, dots, underscores, and hyphens")
    return value


def discover_csvs(inputs: Sequence[Path]) -> list[Path]:
    csvs: set[Path] = set()
    for raw_path in inputs:
        path = raw_path.expanduser().resolve()
        if path.is_dir():
            csvs.update(item.resolve() for item in path.rglob("all_windows.csv") if item.is_file())
        elif path.is_file():
            csvs.add(path)
        else:
            raise FileNotFoundError(f"Input CSV or directory not found: {path}")
    if not csvs:
        raise FileNotFoundError("No all_windows.csv files were found in the supplied inputs")
    return sorted(csvs)


def parse_recording_date(recording: Path) -> date | None:
    match = DATE_PATTERN.search(recording.stem)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%y%m%d").date()
    except ValueError:
        return None


def classify_phase(
    recording_date: date,
    adult_proxy_end: date,
    chick_proxy_start: date,
) -> str | None:
    if adult_proxy_end >= chick_proxy_start:
        raise ValueError("adult_proxy_end must be earlier than chick_proxy_start")
    if recording_date <= adult_proxy_end:
        return "adult_proxy"
    if recording_date >= chick_proxy_start:
        return "chick_proxy"
    return None


def select_proxy_recordings(
    recordings_dir: Path,
    adult_proxy_end: date,
    chick_proxy_start: date,
) -> tuple[list[Path], pd.DataFrame, list[Path]]:
    all_recordings = find_recordings(recordings_dir)
    if not all_recordings:
        raise FileNotFoundError(f"No recursive WAV/FLAC inputs found under {recordings_dir}")
    selected: list[Path] = []
    excluded: list[dict[str, object]] = []
    for recording in all_recordings:
        recording_date = parse_recording_date(recording)
        if recording_date is None:
            excluded.append(
                {
                    "recording": str(recording),
                    "recording_date": "",
                    "exclusion_reason": "unparseable_recording_date",
                }
            )
            continue
        phase = classify_phase(recording_date, adult_proxy_end, chick_proxy_start)
        if phase is None:
            excluded.append(
                {
                    "recording": str(recording),
                    "recording_date": recording_date.isoformat(),
                    "exclusion_reason": "incubation_transition_period",
                }
            )
            continue
        try:
            info = sf.info(recording)
        except (RuntimeError, OSError, ValueError):
            excluded.append(
                {
                    "recording": str(recording),
                    "recording_date": recording_date.isoformat(),
                    "exclusion_reason": "source_audio_unreadable",
                }
            )
            continue
        if int(info.frames) < int(round(WINDOW_SECONDS * info.samplerate)):
            excluded.append(
                {
                    "recording": str(recording),
                    "recording_date": recording_date.isoformat(),
                    "exclusion_reason": "source_audio_shorter_than_window",
                }
            )
            continue
        selected.append(recording)
    excluded_frame = pd.DataFrame(
        excluded,
        columns=["recording", "recording_date", "exclusion_reason"],
    )
    return selected, excluded_frame, all_recordings


def _file_signature(path: Path) -> dict[str, object]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        return {"path": str(resolved), "exists": False}
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "exists": True,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _recording_inventory_signature(recordings_dir: Path, recordings: Sequence[Path]) -> str:
    inventory = []
    for recording in recordings:
        stat = recording.stat()
        inventory.append(
            {
                "relative_path": recording.relative_to(recordings_dir).as_posix(),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    serialized = json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def build_scan_config(
    recordings_dir: Path,
    all_recordings: Sequence[Path],
    selected_recordings: Sequence[Path],
    model_path: Path,
    adult_proxy_end: date,
    chick_proxy_start: date,
) -> dict[str, object]:
    return {
        "scan_config_version": SCAN_CONFIG_VERSION,
        "recordings_dir": str(recordings_dir),
        "recording_count": len(all_recordings),
        "selected_recording_count": len(selected_recordings),
        "recording_inventory_sha256": _recording_inventory_signature(recordings_dir, all_recordings),
        "model": _file_signature(model_path),
        "model_labels": _file_signature(labels_path_for_model(model_path)),
        "adult_proxy_end": adult_proxy_end.isoformat(),
        "chick_proxy_start": chick_proxy_start.isoformat(),
        "sample_rate": SAMPLE_RATE,
        "window_seconds": WINDOW_SECONDS,
        "step_seconds": WINDOW_SECONDS,
    }


def prepare_scan_cache(
    recordings_dir: Path,
    output_dir: Path,
    model_path: Path,
    adult_proxy_end: date,
    chick_proxy_start: date,
    batch_size: int,
    rescan: bool,
    predictor: BatchPredictor | None = None,
) -> tuple[Path, pd.DataFrame, bool]:
    recordings_dir = recordings_dir.expanduser().resolve()
    model_path = model_path.expanduser().resolve()
    selected, excluded_recordings, all_recordings = select_proxy_recordings(
        recordings_dir, adult_proxy_end, chick_proxy_start
    )
    if not selected:
        raise ValueError(
            "No recordings fall in the adult_proxy or chick_proxy date ranges; "
            "check filenames and date boundaries."
        )

    scan_dir = output_dir / "scan"
    scores_path = scan_dir / "all_windows.csv"
    config_path = scan_dir / "scan_config.json"
    excluded_path = scan_dir / "excluded_recordings.csv"
    expected_config = build_scan_config(
        recordings_dir,
        all_recordings,
        selected,
        model_path,
        adult_proxy_end,
        chick_proxy_start,
    )
    cache_complete = scores_path.is_file() and config_path.is_file()
    reused = False

    if cache_complete and not rescan:
        try:
            cached_config = json.loads(config_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise ValueError(f"Invalid scan cache configuration: {config_path}. Use --rescan.") from exc
        if cached_config != expected_config:
            raise ValueError(
                "The recording directory, recording inventory, model, or date boundaries differ "
                "from the saved scan cache. Re-run with --rescan."
            )
        reused = True
        print(f"Reusing comparison scan cache: {scores_path}")
    else:
        scan_dir.mkdir(parents=True, exist_ok=True)
        # A failed rescan must never leave a partial CSV paired with a valid
        # configuration from an earlier completed scan.
        if config_path.exists():
            config_path.unlink()
        scan_recording_paths(
            selected,
            scores_path,
            model_path,
            WINDOW_SECONDS,
            batch_size,
            force=scores_path.exists(),
            predictor=predictor,
        )
        config_path.write_text(
            json.dumps(expected_config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"Saved comparison scan cache: {scores_path}")

    scan_dir.mkdir(parents=True, exist_ok=True)
    excluded_recordings.to_csv(excluded_path, index=False)
    return scores_path, excluded_recordings, reused


def load_audio_window(recording: Path, start_seconds: float) -> np.ndarray:
    with sf.SoundFile(recording) as source:
        source_rate = int(source.samplerate)
        start_frame = int(round(start_seconds * source_rate))
        window_frames = int(round(WINDOW_SECONDS * source_rate))
        if start_frame < 0 or start_frame + window_frames > len(source):
            raise ValueError("source_audio_shorter_than_window")
        source.seek(start_frame)
        audio = source.read(window_frames, dtype="float32", always_2d=True)
    mono = np.asarray(audio.mean(axis=1), dtype=np.float32)
    if source_rate != SAMPLE_RATE:
        factor = gcd(source_rate, SAMPLE_RATE)
        mono = resample_poly(mono, SAMPLE_RATE // factor, source_rate // factor).astype(np.float32)
    if len(mono) < SAMPLE_LENGTH:
        raise ValueError("source_audio_shorter_than_window_after_resampling")
    return np.asarray(mono[:SAMPLE_LENGTH], dtype=np.float32)


def compute_features(audio: np.ndarray) -> tuple[dict[str, float], np.ndarray]:
    audio = np.asarray(audio, dtype=np.float32)
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    magnitude = np.abs(librosa.stft(audio, n_fft=2048, hop_length=512, center=False))
    power = np.square(magnitude, dtype=np.float64)
    spectrum = np.mean(power, axis=1)
    frequencies = np.fft.rfftfreq(2048, d=1.0 / SAMPLE_RATE)
    peak_frequency = float(frequencies[int(np.argmax(spectrum))]) if np.any(spectrum) else 0.0
    total_power = float(spectrum.sum())
    if total_power > 0:
        distribution = spectrum / total_power
        positive = distribution > 0
        entropy = float(
            -np.sum(distribution[positive] * np.log(distribution[positive])) / np.log(len(distribution))
        )
    else:
        entropy = 0.0
    mel_power = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=2048,
        hop_length=512,
        n_mels=128,
        fmin=0,
        fmax=SAMPLE_RATE / 2,
        power=2.0,
        center=False,
    )
    log_mel = librosa.power_to_db(mel_power, ref=1.0, top_db=100.0)
    features = {
        "rms": rms,
        "dbfs": float(20.0 * np.log10(max(rms, np.finfo(float).tiny))),
        "peak_frequency_hz": peak_frequency,
        "spectral_centroid_hz": float(np.mean(librosa.feature.spectral_centroid(S=magnitude, sr=SAMPLE_RATE))),
        "spectral_bandwidth_hz": float(
            np.mean(librosa.feature.spectral_bandwidth(S=magnitude, sr=SAMPLE_RATE))
        ),
        "rolloff_85_hz": float(
            np.mean(librosa.feature.spectral_rolloff(S=magnitude, sr=SAMPLE_RATE, roll_percent=0.85))
        ),
        "spectral_flatness": float(np.mean(librosa.feature.spectral_flatness(S=magnitude))),
        "spectral_entropy": entropy,
        "zero_crossing_rate": float(
            np.mean(librosa.feature.zero_crossing_rate(audio, frame_length=2048, hop_length=512, center=False))
        ),
    }
    return features, np.asarray(log_mel, dtype=np.float32)


def _excluded_row(row: dict[str, object], reason: str) -> dict[str, object]:
    return {
        "input_csv": row.get("input_csv", ""),
        "recording": row.get("recording", ""),
        "start_seconds": row.get("start_seconds", ""),
        "end_seconds": row.get("end_seconds", ""),
        "probability": row.get("probability", ""),
        "exclusion_reason": reason,
    }


def validate_windows(
    csv_paths: Sequence[Path],
    threshold: float,
    adult_proxy_end: date,
    chick_proxy_start: date,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, list[np.ndarray]]]:
    """Return high-confidence feature rows, all eligible rows, exclusions, and log-mels."""
    included: list[dict[str, object]] = []
    eligible: list[dict[str, object]] = []
    excluded: list[dict[str, object]] = []
    seen: set[tuple[str, int]] = set()
    duration_cache: dict[Path, float] = {}
    frame_cache: dict[Path, tuple[int, int]] = {}
    log_mels: dict[str, list[np.ndarray]] = {phase: [] for phase in PHASES}

    for csv_path in csv_paths:
        frame = pd.read_csv(csv_path)
        missing = REQUIRED_COLUMNS - set(frame.columns)
        if missing:
            raise ValueError(f"{csv_path} is missing required columns: {sorted(missing)}")
        for raw in frame.to_dict(orient="records"):
            raw["input_csv"] = str(csv_path)
            try:
                probability = float(raw["probability"])
                start = float(raw["start_seconds"])
                end = float(raw["end_seconds"])
            except (TypeError, ValueError):
                excluded.append(_excluded_row(raw, "invalid_numeric_value"))
                continue
            if not np.isfinite(probability) or not np.isfinite(start) or not np.isfinite(end):
                excluded.append(_excluded_row(raw, "invalid_numeric_value"))
                continue
            recording = Path(str(raw["recording"])).expanduser().resolve()
            key = (str(recording), round(start * SAMPLE_RATE))
            if key in seen:
                excluded.append(_excluded_row(raw, "duplicate_recording_start"))
                continue
            seen.add(key)
            recording_date = parse_recording_date(recording)
            if recording_date is None:
                excluded.append(_excluded_row(raw, "unparseable_recording_date"))
                continue
            phase = classify_phase(recording_date, adult_proxy_end, chick_proxy_start)
            if phase is None:
                excluded.append(_excluded_row(raw, "incubation_transition_period"))
                continue
            if start < 0 or abs((end - start) - WINDOW_SECONDS) > 1e-6:
                excluded.append(_excluded_row(raw, "not_complete_three_second_window"))
                continue
            if not recording.is_file():
                excluded.append(_excluded_row(raw, "source_file_missing"))
                continue
            try:
                if recording not in frame_cache:
                    info = sf.info(recording)
                    frame_cache[recording] = (int(info.frames), int(info.samplerate))
                    duration_cache[recording] = info.frames / info.samplerate
                source_frames, source_rate = frame_cache[recording]
            except (RuntimeError, OSError, ValueError):
                excluded.append(_excluded_row(raw, "source_audio_unreadable"))
                continue
            start_frame = int(round(start * source_rate))
            needed_frames = int(round(WINDOW_SECONDS * source_rate))
            if start_frame < 0 or start_frame + needed_frames > source_frames:
                excluded.append(_excluded_row(raw, "source_audio_shorter_than_window"))
                continue

            base = {
                "input_csv": str(csv_path),
                "recording": str(recording),
                "recording_date": recording_date.isoformat(),
                "phase": phase,
                "start_seconds": start,
                "end_seconds": end,
                "probability": probability,
                "recording_duration_seconds": duration_cache[recording],
            }
            eligible.append(base)
            if probability < threshold:
                excluded.append(_excluded_row(raw, "below_analysis_threshold"))
                continue
            try:
                audio = load_audio_window(recording, start)
            except ValueError as exc:
                excluded.append(_excluded_row(raw, str(exc) or "source_audio_read_error"))
                eligible.pop()
                continue
            except (RuntimeError, OSError):
                excluded.append(_excluded_row(raw, "source_audio_read_error"))
                eligible.pop()
                continue
            try:
                features, log_mel = compute_features(audio)
            except (RuntimeError, ValueError, FloatingPointError):
                excluded.append(_excluded_row(raw, "feature_extraction_error"))
                eligible.pop()
                continue
            included.append({**base, **features})
            log_mels[phase].append(log_mel)

    included_frame = pd.DataFrame(included, columns=ANALYSIS_COLUMNS)
    eligible_frame = pd.DataFrame(eligible, columns=ELIGIBLE_COLUMNS)
    excluded_frame = pd.DataFrame(
        excluded,
        columns=["input_csv", "recording", "start_seconds", "end_seconds", "probability", "exclusion_reason"],
    )
    return included_frame, eligible_frame, excluded_frame, log_mels
