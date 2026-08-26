"""使用成鸟期与雏鸟期两个 TFLite 模型扫描相同长录音窗口。"""

# ============================================================
# dual_model_scan.py
# 用成鸟模型和雏鸟模型同时对长录音进行推理，
# 将两个模型的分数按窗口精确合并。
# ============================================================
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from project_config import CHICK_SCREENING_DIR, print_config

SAMPLE_RATE = 48_000
WINDOW_SECONDS = 3.0
SAMPLE_LENGTH = int(SAMPLE_RATE * WINDOW_SECONDS)
POSITIVE_CLASS = "storm_petrel"
AUDIO_SUFFIXES = {".wav", ".flac"}

DUAL_CSV_FIELDS = [
    "recording",
    "start_seconds",
    "end_seconds",
    "logit_adult",
    "prob_adult",
    "logit_chick",
    "prob_chick",
    "delta_prob",
    "delta_logit",
]


def sigmoid(value: float) -> float:
    """以数值稳定方式把单个 logit 转为 sigmoid 分数。"""

    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


class SingleModelPredictor:
    """加载单个 TFLite 模型，返回 storm_petrel 的 raw logit 列。"""

    def __init__(self, model_path: Path) -> None:
        if not model_path.is_file():
            raise FileNotFoundError(f"Model not found: {model_path}")
        labels_path = model_path.with_name(f"{model_path.stem}_Labels.txt")
        if not labels_path.is_file():
            raise FileNotFoundError(f"Labels not found: {labels_path}")
        labels = [
            line.strip()
            for line in labels_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if POSITIVE_CLASS not in labels:
            raise ValueError(
                f"{POSITIVE_CLASS!r} not in model labels: {labels_path}"
            )
        self.positive_index = labels.index(POSITIVE_CLASS)

        from tensorflow import lite as tflite

        self.interpreter = tflite.Interpreter(model_path=str(model_path))
        self.interpreter.allocate_tensors()
        self.input_detail = self.interpreter.get_input_details()[0]
        self.output_detail = self.interpreter.get_output_details()[0]
        signature = self.input_detail.get(
            "shape_signature", self.input_detail["shape"]
        )
        if len(signature) != 2 or int(signature[1]) != SAMPLE_LENGTH:
            raise ValueError(
                f"Expected model input [batch, {SAMPLE_LENGTH}], "
                f"got {signature.tolist()}"
            )

    def predict(self, batch: np.ndarray) -> np.ndarray:
        """执行单个日期模型的批量 TFLite 推理。"""

        batch = np.asarray(batch, dtype=np.float32)
        self.interpreter.resize_tensor_input(
            self.input_detail["index"], list(batch.shape)
        )
        self.interpreter.allocate_tensors()
        self.input_detail = self.interpreter.get_input_details()[0]
        self.output_detail = self.interpreter.get_output_details()[0]
        self.interpreter.set_tensor(self.input_detail["index"], batch)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_detail["index"])
        return output[:, self.positive_index].astype(np.float64)


class DualModelPredictor:
    """封装两个 TFLite 模型，一次 predict() 返回两列 logit。"""

    def __init__(self, adult_model_path: Path, chick_model_path: Path) -> None:
        self.adult = SingleModelPredictor(adult_model_path)
        self.chick = SingleModelPredictor(chick_model_path)

    def predict(self, batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """对同一批窗口返回成鸟和雏鸟模型 logits。"""

        return self.adult.predict(batch), self.chick.predict(batch)


def find_recordings(input_root: Path) -> list[Path]:
    """递归发现支持的录音文件并确定性排序。"""

    if not input_root.is_dir():
        raise NotADirectoryError(f"Input directory not found: {input_root}")
    return sorted(
        p.resolve()
        for p in input_root.rglob("*")
        if p.is_file() and p.suffix.lower() in AUDIO_SUFFIXES
    )


def iter_audio_windows(path: Path):
    """生成录音中每个完整 3 秒窗口的 (start_seconds, audio_array)。"""
    import soundfile as sf
    from math import gcd
    from scipy.signal import resample_poly

    with sf.SoundFile(path) as source:
        source_rate = int(source.samplerate)
        total_frames = len(source)
        window_frames = int(round(WINDOW_SECONDS * source_rate))
        step_frames = window_frames  # non-overlapping
        start_frame = 0
        while start_frame + window_frames <= total_frames:
            source.seek(start_frame)
            audio = source.read(window_frames, dtype="float32", always_2d=True)
            if audio.ndim == 2:
                audio = audio.mean(axis=1)
            audio = np.asarray(audio, dtype=np.float32)
            if source_rate != SAMPLE_RATE and audio.size:
                factor = gcd(source_rate, SAMPLE_RATE)
                audio = resample_poly(
                    audio, SAMPLE_RATE // factor, source_rate // factor
                ).astype(np.float32)
            if len(audio) < SAMPLE_LENGTH:
                audio = np.pad(audio, (0, SAMPLE_LENGTH - len(audio)))
            yield (
                start_frame / source_rate,
                (start_frame + window_frames) / source_rate,
                np.asarray(audio[:SAMPLE_LENGTH], dtype=np.float32),
            )
            start_frame += step_frames


def dual_scan(
    recordings: Sequence[Path],
    output_csv: Path,
    adult_model_path: Path,
    chick_model_path: Path,
    batch_size: int = 32,
    force: bool = False,
) -> int:
    """扫描全部录音并写出两个模型的逐窗口分数。"""

    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    if not recordings:
        raise FileNotFoundError("No recordings supplied for scanning")

    resolved = sorted({p.expanduser().resolve() for p in recordings})
    missing = [p for p in resolved if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Recording not found: {missing[0]}")

    if output_csv.exists() and not force:
        raise FileExistsError(
            f"Scores already exist: {output_csv}. Use --force to replace."
        )
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    predictor = DualModelPredictor(adult_model_path, chick_model_path)
    count = 0
    batch_starts: list[float] = []
    batch_ends: list[float] = []
    batch_recording: str = ""
    batch_audio: list[np.ndarray] = []

    def flush() -> int:
        """对当前批次推理、写行并清空缓冲区。"""

        nonlocal count
        if not batch_audio:
            return 0
        logits_adult, logits_chick = predictor.predict(
            np.stack(batch_audio)
        )
        written = 0
        for i in range(len(batch_audio)):
            la = float(logits_adult[i])
            lc = float(logits_chick[i])
            pa = sigmoid(la)
            pc = sigmoid(lc)
            writer.writerow({
                "recording": batch_recording,
                "start_seconds": f"{batch_starts[i]:.6f}",
                "end_seconds": f"{batch_ends[i]:.6f}",
                "logit_adult": f"{la:.8f}",
                "prob_adult": f"{pa:.8f}",
                "logit_chick": f"{lc:.8f}",
                "prob_chick": f"{pc:.8f}",
                "delta_prob": f"{pc - pa:.8f}",
                "delta_logit": f"{lc - la:.8f}",
            })
            written += 1
        count += written
        batch_audio.clear()
        batch_starts.clear()
        batch_ends.clear()
        return written

    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DUAL_CSV_FIELDS)
        writer.writeheader()
        for recording in resolved:
            print(f"Scanning {recording}")
            batch_recording = str(recording)
            for start, end, audio in iter_audio_windows(recording):
                batch_starts.append(start)
                batch_ends.append(end)
                batch_audio.append(audio)
                if len(batch_audio) == batch_size:
                    flush()
            flush()

    print(f"Wrote {count} dual-model window scores to {output_csv}")
    return count


def parse_args() -> argparse.Namespace:
    """解析双模型扫描输入、模型、批量大小和输出选项。"""

    parser = argparse.ArgumentParser(
        description="Dual-model scan: adult and chick models on the same recordings."
    )
    parser.add_argument("input_dir", type=Path)
    parser.add_argument(
        "--adult-model",
        type=Path,
        default=CHICK_SCREENING_DIR / "models" / "adult_model" / "model.tflite",
    )
    parser.add_argument(
        "--chick-model",
        type=Path,
        default=CHICK_SCREENING_DIR / "models" / "chick_model" / "model.tflite",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Defaults to workspace/chick_screening/dual_scores.csv",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    """运行双模型长录音扫描命令行入口。"""

    args = parse_args()
    output_csv = (
        args.output_csv.expanduser().resolve()
        if args.output_csv
        else CHICK_SCREENING_DIR / "dual_scores.csv"
    )
    print_config()
    recordings = find_recordings(args.input_dir.expanduser().resolve())
    print(f"Found {len(recordings)} recordings")
    dual_scan(
        recordings,
        output_csv,
        args.adult_model.expanduser().resolve(),
        args.chick_model.expanduser().resolve(),
        args.batch_size,
        args.force,
    )


if __name__ == "__main__":
    main()
