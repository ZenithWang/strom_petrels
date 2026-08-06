from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
import sys
from dataclasses import dataclass
from math import gcd
from pathlib import Path
from typing import Iterator

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from project_config import print_config

SAMPLE_RATE = 48_000
DEFAULT_THRESHOLD = 0.20
DETECTION_FIELDS = [
    "recording",
    "hit_start_seconds",
    "hit_end_seconds",
    "start_seconds",
    "end_seconds",
    "max_probability",
    "mean_probability",
    "window_count",
    "candidate_path",
]


@dataclass
class HitGroup:
    recording: Path
    hit_start_seconds: float
    hit_end_seconds: float
    max_probability: float
    probability_sum: float
    window_count: int


def iter_hit_groups(scores_csv: Path, threshold: float) -> Iterator[HitGroup]:
    """Yield every thresholded three-second window as an independent hit."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    if not scores_csv.is_file():
        raise FileNotFoundError(f"Window score CSV not found: {scores_csv}")
    with scores_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"recording", "start_seconds", "end_seconds", "probability"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing all_windows.csv fields: {sorted(missing)}")
        for row in reader:
            probability = float(row["probability"])
            if probability < threshold:
                continue
            recording = Path(row["recording"]).expanduser().resolve()
            start_seconds = float(row["start_seconds"])
            end_seconds = float(row["end_seconds"])
            if abs((end_seconds - start_seconds) - 3.0) > 1e-6:
                raise ValueError(
                    f"Expected a complete 3-second window, got "
                    f"{start_seconds:.6f}-{end_seconds:.6f} in {scores_csv}"
                )
            yield HitGroup(recording, start_seconds, end_seconds, probability, probability, 1)


def read_candidate_audio(path: Path, start_seconds: float, end_seconds: float) -> np.ndarray:
    with sf.SoundFile(path) as source:
        source_rate = int(source.samplerate)
        start_frame = max(0, int(round(start_seconds * source_rate)))
        end_frame = min(len(source), int(round(end_seconds * source_rate)))
        source.seek(start_frame)
        audio = source.read(max(0, end_frame - start_frame), dtype="float32", always_2d=True)
    mono = audio.mean(axis=1).astype(np.float32)
    if source_rate != SAMPLE_RATE and mono.size:
        factor = gcd(source_rate, SAMPLE_RATE)
        mono = resample_poly(mono, SAMPLE_RATE // factor, source_rate // factor).astype(np.float32)
    return mono


def candidate_name(recording: Path, index: int, start_seconds: float, end_seconds: float) -> str:
    digest = hashlib.sha1(str(recording).encode("utf-8")).hexdigest()[:10]
    safe_stem = "".join(character if character.isalnum() or character in "-_" else "_" for character in recording.stem)
    start_ms = round(start_seconds * 1_000)
    end_ms = round(end_seconds * 1_000)
    return f"{safe_stem}_{digest}_{index:04d}_{start_ms:010d}_{end_ms:010d}.wav"


def extract_candidates(
    scores_csv: Path,
    detections_csv: Path,
    candidates_dir: Path,
    threshold: float = DEFAULT_THRESHOLD,
    force: bool = False,
) -> int:
    if (detections_csv.exists() or candidates_dir.exists()) and not force:
        raise FileExistsError(
            f"Detection outputs already exist ({detections_csv}, {candidates_dir}). Use --force to replace them."
        )
    if candidates_dir.exists():
        shutil.rmtree(candidates_dir)
    if detections_csv.exists():
        detections_csv.unlink()
    candidates_dir.mkdir(parents=True, exist_ok=True)
    detections_csv.parent.mkdir(parents=True, exist_ok=True)

    detection_count = 0
    recording_indices: dict[Path, int] = {}
    with detections_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DETECTION_FIELDS)
        writer.writeheader()
        for group in iter_hit_groups(scores_csv, threshold):
            recording = group.recording
            if not recording.is_file():
                raise FileNotFoundError(f"Source recording from scores CSV no longer exists: {recording}")
            index = recording_indices.get(recording, 0) + 1
            recording_indices[recording] = index
            hit_start = group.hit_start_seconds
            hit_end = group.hit_end_seconds
            start = hit_start
            end = hit_end
            name = candidate_name(recording, index, start, end)
            candidate_path = candidates_dir / name
            audio = read_candidate_audio(recording, start, end)
            if len(audio) != int(SAMPLE_RATE * 3.0):
                raise ValueError(
                    f"Source recording does not contain the complete three-second hit "
                    f"{start:.6f}-{end:.6f}: {recording}"
                )
            sf.write(candidate_path, audio, SAMPLE_RATE, subtype="PCM_16")
            writer.writerow(
                {
                    "recording": str(recording),
                    "hit_start_seconds": f"{hit_start:.6f}",
                    "hit_end_seconds": f"{hit_end:.6f}",
                    "start_seconds": f"{start:.6f}",
                    "end_seconds": f"{end:.6f}",
                    "max_probability": f"{group.max_probability:.8f}",
                    "mean_probability": f"{group.probability_sum / group.window_count:.8f}",
                    "window_count": group.window_count,
                    "candidate_path": str(candidate_path.resolve()),
                }
            )
            detection_count += 1
    print(f"Wrote {detection_count} independent three-second detections to {detections_csv}")
    print(f"Candidate WAV files: {candidates_dir}")
    return detection_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract each thresholded three-second window as a candidate WAV.")
    parser.add_argument("all_windows_csv", type=Path)
    parser.add_argument("--detections-csv", type=Path)
    parser.add_argument("--candidates-dir", type=Path)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scores_csv = args.all_windows_csv.expanduser().resolve()
    run_dir = scores_csv.parent
    detections_csv = (args.detections_csv or run_dir / "detections.csv").expanduser().resolve()
    candidates_dir = (args.candidates_dir or run_dir / "candidates").expanduser().resolve()
    print_config()
    extract_candidates(
        scores_csv,
        detections_csv,
        candidates_dir,
        args.threshold,
        args.force,
    )


if __name__ == "__main__":
    main()
