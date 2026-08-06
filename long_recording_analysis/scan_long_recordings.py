from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from math import gcd
from pathlib import Path
from typing import Iterator, Protocol, Sequence

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from project_config import DEPLOYMENT_MODEL_DIR, print_config

SAMPLE_RATE = 48_000
WINDOW_SECONDS = 3.0
SAMPLE_LENGTH = int(SAMPLE_RATE * WINDOW_SECONDS)
DEFAULT_STEP_SECONDS = WINDOW_SECONDS
POSITIVE_CLASS = "storm_petrel"
AUDIO_SUFFIXES = {".wav", ".flac"}
CSV_FIELDS = ["recording", "start_seconds", "end_seconds", "raw_logit", "probability"]


@dataclass(frozen=True)
class AudioWindow:
    recording: Path
    start_seconds: float
    end_seconds: float
    audio: np.ndarray


class BatchPredictor(Protocol):
    def predict(self, batch: np.ndarray) -> np.ndarray: ...


def sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def labels_path_for_model(model_path: Path) -> Path:
    return model_path.with_name(f"{model_path.stem}_Labels.txt")


class TFLitePredictor:
    def __init__(self, model_path: Path) -> None:
        if not model_path.is_file():
            raise FileNotFoundError(f"Deployment model not found: {model_path}")
        labels_path = labels_path_for_model(model_path)
        if not labels_path.is_file():
            raise FileNotFoundError(f"Model labels not found: {labels_path}")
        self.labels = [line.strip() for line in labels_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if POSITIVE_CLASS not in self.labels:
            raise ValueError(f"{POSITIVE_CLASS!r} is not in {labels_path}: {self.labels}")
        self.positive_index = self.labels.index(POSITIVE_CLASS)

        from tensorflow import lite as tflite

        self.interpreter = tflite.Interpreter(model_path=str(model_path))
        self.interpreter.allocate_tensors()
        self.input_detail = self.interpreter.get_input_details()[0]
        self.output_detail = self.interpreter.get_output_details()[0]
        signature = self.input_detail.get("shape_signature", self.input_detail["shape"])
        if len(signature) != 2 or int(signature[1]) != SAMPLE_LENGTH:
            raise ValueError(f"Expected model input [batch, {SAMPLE_LENGTH}], got {signature.tolist()}")
        if self.input_detail["dtype"] != np.float32:
            raise ValueError(f"Expected float32 model input, got {self.input_detail['dtype']}")

    def predict(self, batch: np.ndarray) -> np.ndarray:
        batch = np.asarray(batch, dtype=np.float32)
        if batch.ndim != 2 or batch.shape[1] != SAMPLE_LENGTH:
            raise ValueError(f"Expected batch shape [n, {SAMPLE_LENGTH}], got {batch.shape}")
        self.interpreter.resize_tensor_input(self.input_detail["index"], list(batch.shape))
        self.interpreter.allocate_tensors()
        self.input_detail = self.interpreter.get_input_details()[0]
        self.output_detail = self.interpreter.get_output_details()[0]
        self.interpreter.set_tensor(self.input_detail["index"], batch)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_detail["index"])
        if output.ndim != 2 or output.shape[1] != len(self.labels):
            raise ValueError(f"Model output shape {output.shape} does not match {len(self.labels)} labels")
        return output[:, self.positive_index].astype(np.float64)


def find_recordings(input_root: Path) -> list[Path]:
    if not input_root.is_dir():
        raise NotADirectoryError(f"Long-recording input directory not found: {input_root}")
    return sorted(
        path.resolve()
        for path in input_root.rglob("*")
        if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES
    )


def _resample_window(audio: np.ndarray, source_rate: int) -> np.ndarray:
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = np.asarray(audio, dtype=np.float32)
    if source_rate != SAMPLE_RATE and audio.size:
        factor = gcd(source_rate, SAMPLE_RATE)
        audio = resample_poly(audio, SAMPLE_RATE // factor, source_rate // factor).astype(np.float32)
    if len(audio) < SAMPLE_LENGTH:
        audio = np.pad(audio, (0, SAMPLE_LENGTH - len(audio)))
    return np.asarray(audio[:SAMPLE_LENGTH], dtype=np.float32)


def iter_audio_windows(path: Path, step_seconds: float = DEFAULT_STEP_SECONDS) -> Iterator[AudioWindow]:
    if step_seconds <= 0:
        raise ValueError("step_seconds must be greater than zero")
    with sf.SoundFile(path) as source:
        source_rate = int(source.samplerate)
        total_frames = len(source)
        window_frames = int(round(WINDOW_SECONDS * source_rate))
        step_frames = max(1, int(round(step_seconds * source_rate)))
        start_frame = 0
        # Only score complete windows. A short tail is not model input and must
        # not be padded into an artificial detection.
        while start_frame + window_frames <= total_frames:
            source.seek(start_frame)
            audio = source.read(window_frames, dtype="float32", always_2d=True)
            yield AudioWindow(
                recording=path.resolve(),
                start_seconds=start_frame / source_rate,
                end_seconds=(start_frame + window_frames) / source_rate,
                audio=_resample_window(audio, source_rate),
            )
            start_frame += step_frames


def _write_batch(writer: csv.DictWriter, windows: list[AudioWindow], predictor: BatchPredictor) -> int:
    if not windows:
        return 0
    logits = predictor.predict(np.stack([window.audio for window in windows]))
    if len(logits) != len(windows):
        raise ValueError(f"Predictor returned {len(logits)} scores for {len(windows)} windows")
    for window, raw_logit in zip(windows, logits):
        probability = sigmoid(float(raw_logit))
        writer.writerow(
            {
                "recording": str(window.recording),
                "start_seconds": f"{window.start_seconds:.6f}",
                "end_seconds": f"{window.end_seconds:.6f}",
                "raw_logit": f"{float(raw_logit):.8f}",
                "probability": f"{probability:.8f}",
            }
        )
    return len(windows)


def scan_recording_paths(
    recordings: Sequence[Path],
    output_csv: Path,
    model_path: Path,
    step_seconds: float = DEFAULT_STEP_SECONDS,
    batch_size: int = 32,
    force: bool = False,
    predictor: BatchPredictor | None = None,
) -> int:
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    if not recordings:
        raise FileNotFoundError("No WAV/FLAC recordings were supplied for scanning")
    resolved_recordings = sorted({path.expanduser().resolve() for path in recordings})
    missing = [path for path in resolved_recordings if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Recording selected for scanning does not exist: {missing[0]}")
    if output_csv.exists() and not force:
        raise FileExistsError(f"Window scores already exist: {output_csv}. Use --force to replace them.")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    predictor = predictor or TFLitePredictor(model_path)
    count = 0
    batch: list[AudioWindow] = []
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for recording in resolved_recordings:
            print(f"Scanning {recording}")
            for window in iter_audio_windows(recording, step_seconds):
                batch.append(window)
                if len(batch) == batch_size:
                    count += _write_batch(writer, batch, predictor)
                    batch.clear()
        count += _write_batch(writer, batch, predictor)
    print(f"Wrote {count} window scores to {output_csv}")
    return count


def scan_recordings(
    input_root: Path,
    output_csv: Path,
    model_path: Path,
    step_seconds: float = DEFAULT_STEP_SECONDS,
    batch_size: int = 32,
    force: bool = False,
    predictor: BatchPredictor | None = None,
) -> int:
    recordings = find_recordings(input_root)
    if not recordings:
        raise FileNotFoundError(f"No recursive WAV/FLAC inputs found under {input_root}")
    return scan_recording_paths(
        recordings,
        output_csv,
        model_path,
        step_seconds,
        batch_size,
        force,
        predictor,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream WAV/FLAC recordings through a deployment TFLite model.")
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEPLOYMENT_MODEL_DIR / "model.tflite")
    parser.add_argument("--step", type=float, default=DEFAULT_STEP_SECONDS, help="Window step in seconds.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print_config()
    scan_recordings(
        args.input_dir.expanduser().resolve(),
        args.output_csv.expanduser().resolve(),
        args.model.expanduser().resolve(),
        args.step,
        args.batch_size,
        args.force,
    )


if __name__ == "__main__":
    main()
