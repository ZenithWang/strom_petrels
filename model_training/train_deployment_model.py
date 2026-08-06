from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from project_config import (
    CLIPS_DIR,
    DEPLOYMENT_MODEL_DIR,
    NUMBA_CACHE_DIR,
    PROJECT_ROOT,
    TMP_DIR,
    print_config,
)

CLASS_NAMES = ("storm_petrel", "not_storm_petrel")
EXPECTED_PER_CLASS = 1_000
LEARNING_RATE = "0.0001"
HIDDEN_UNITS = "256"
DROPOUT = "0.3"
VALIDATION_SPLIT = "0.2"
BATCH_SIZE = "32"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the all-clips BirdNET deployment classifier.")
    parser.add_argument("--epochs", type=int, default=os.environ.get("BIRDNET_EPOCHS", "50"))
    parser.add_argument("--threads", type=int, default=os.environ.get("BIRDNET_THREADS", "8"))
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete and replace an existing deployment output directory.",
    )
    return parser.parse_args()


def configure_cache() -> None:
    NUMBA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["NUMBA_CACHE_DIR"] = str(NUMBA_CACHE_DIR)
    os.environ["TMP"] = str(TMP_DIR)
    os.environ["TEMP"] = str(TMP_DIR)


def count_training_clips(clips_dir: Path = CLIPS_DIR) -> dict[str, int]:
    counts: dict[str, int] = {}
    for class_name in CLASS_NAMES:
        class_dir = clips_dir / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing class directory: {class_dir}")
        counts[class_name] = sum(1 for path in class_dir.iterdir() if path.is_file() and path.suffix.lower() == ".wav")
    unexpected = sorted(path.name for path in clips_dir.iterdir() if path.is_dir() and path.name not in CLASS_NAMES)
    if unexpected:
        raise ValueError(f"Unexpected class directories under {clips_dir}: {unexpected}")
    wrong = {name: count for name, count in counts.items() if count != EXPECTED_PER_CLASS}
    if wrong:
        raise ValueError(
            f"Deployment training requires exactly {EXPECTED_PER_CLASS} WAV files per class; found {counts}"
        )
    return counts


def training_command(epochs: int, threads: int) -> list[str]:
    output_prefix = DEPLOYMENT_MODEL_DIR / "model"
    cache_file = DEPLOYMENT_MODEL_DIR / "training_cache.npz"
    return [
        sys.executable,
        "-m",
        "birdnet_analyzer.train",
        str(CLIPS_DIR),
        "-o",
        str(output_prefix),
        "--epochs",
        str(epochs),
        "--threads",
        str(threads),
        "--batch_size",
        BATCH_SIZE,
        "--val_split",
        VALIDATION_SPLIT,
        "--learning_rate",
        LEARNING_RATE,
        "--hidden_units",
        HIDDEN_UNITS,
        "--dropout",
        DROPOUT,
        "--model_save_mode",
        "replace",
        "--model_format",
        "tflite",
        "--cache_mode",
        "save",
        "--cache_file",
        str(cache_file),
    ]


def validate_artifacts(output_dir: Path = DEPLOYMENT_MODEL_DIR) -> None:
    required = [
        output_dir / "model.tflite",
        output_dir / "model_Labels.txt",
        output_dir / "model_Params.csv",
        output_dir / "model_sample_counts.csv",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Training completed without required deployment artifacts: {missing}")


def train(epochs: int, threads: int, force: bool = False) -> None:
    if not 1 <= epochs <= 50:
        raise ValueError("epochs must be between 1 and the configured maximum of 50")
    if threads <= 0:
        raise ValueError("threads must be greater than zero")
    counts = count_training_clips()
    if DEPLOYMENT_MODEL_DIR.exists():
        if not force:
            raise FileExistsError(
                f"Deployment output already exists: {DEPLOYMENT_MODEL_DIR}. Use --force to replace it."
            )
        shutil.rmtree(DEPLOYMENT_MODEL_DIR)
    DEPLOYMENT_MODEL_DIR.mkdir(parents=True)

    command = training_command(epochs, threads)
    config = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(CLIPS_DIR),
        "output_dir": str(DEPLOYMENT_MODEL_DIR),
        "class_counts": counts,
        "total_clips": sum(counts.values()),
        "validation_split": float(VALIDATION_SPLIT),
        "learning_rate": float(LEARNING_RATE),
        "hidden_units": int(HIDDEN_UNITS),
        "dropout": float(DROPOUT),
        "max_epochs": int(epochs),
        "batch_size": int(BATCH_SIZE),
        "threads": int(threads),
        "command": command,
    }
    (DEPLOYMENT_MODEL_DIR / "training_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Training clip counts: {counts}")
    print("Running training command:")
    print(" ".join(command))
    result = subprocess.run(command, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        raise RuntimeError(f"BirdNET deployment training failed with return code {result.returncode}")
    validate_artifacts(DEPLOYMENT_MODEL_DIR)
    print(f"Deployment model written to {DEPLOYMENT_MODEL_DIR}")


def main() -> None:
    args = parse_args()
    configure_cache()
    print_config()
    print(f"Epochs: {args.epochs}; threads: {args.threads}; batch size: {BATCH_SIZE}")
    train(args.epochs, args.threads, args.force)


if __name__ == "__main__":
    main()
