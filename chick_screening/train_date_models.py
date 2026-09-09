"""分别训练成鸟代理期与雏鸟代理期的 BirdNET 分类模型。"""

# ============================================================
# train_date_models.py
# 分别用 adult_clips 和 chick_clips 训练 BirdNET 二分类模型。
# ============================================================
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
    CHICK_SCREENING_DIR,
    NUMBA_CACHE_DIR,
    PROJECT_ROOT,
    TMP_DIR,
    print_config,
)

CLASS_NAMES = ("storm_petrel", "not_storm_petrel")
LEARNING_RATE = "0.0001"
HIDDEN_UNITS = "256"
DROPOUT = "0.3"
VALIDATION_SPLIT = "0.2"
BATCH_SIZE = "32"


def configure_cache() -> None:
    """把训练缓存和临时文件固定到 workspace。"""

    NUMBA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["NUMBA_CACHE_DIR"] = str(NUMBA_CACHE_DIR)
    os.environ["TMP"] = str(TMP_DIR)
    os.environ["TEMP"] = str(TMP_DIR)


def count_training_clips(clips_dir: Path) -> dict[str, int]:
    """验证日期数据集的类别目录并统计 WAV 数量。"""

    counts: dict[str, int] = {}
    for class_name in CLASS_NAMES:
        class_dir = clips_dir / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing class directory: {class_dir}")
        counts[class_name] = sum(
            1 for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() == ".wav"
        )
    unexpected = sorted(
        p.name for p in clips_dir.iterdir()
        if p.is_dir() and p.name not in CLASS_NAMES
    )
    if unexpected:
        raise ValueError(f"Unexpected class directories under {clips_dir}: {unexpected}")
    return counts


def training_command(
    clips_dir: Path,
    output_dir: Path,
    epochs: int,
    threads: int,
) -> list[str]:
    """构建一个日期模型的 BirdNET-Analyzer 训练命令。"""

    output_prefix = output_dir / "model"
    cache_file = output_dir / "training_cache.npz"
    return [
        sys.executable,
        "-m",
        "birdnet_analyzer.train",
        str(clips_dir),
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


def validate_artifacts(output_dir: Path) -> None:
    """确认日期模型训练生成了全部必需产物。"""

    required = [
        output_dir / "model.tflite",
        output_dir / "model_Labels.txt",
        output_dir / "model_Params.csv",
        output_dir / "model_sample_counts.csv",
    ]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise RuntimeError(
            f"Training completed without required artifacts: {missing}"
        )


def train_one_model(
    clips_dir: Path,
    output_dir: Path,
    epochs: int,
    threads: int,
    force: bool = False,
) -> None:
    """训练一个时期的模型并保存可追踪配置。"""

    if not 1 <= epochs <= 50:
        raise ValueError("epochs must be between 1 and 50")
    if threads <= 0:
        raise ValueError("threads must be greater than zero")

    counts = count_training_clips(clips_dir)
    total = sum(counts.values())
    if total == 0:
        raise ValueError(f"No WAV files found under {clips_dir}")

    if output_dir.exists():
        if not force:
            raise FileExistsError(
                f"Model output already exists: {output_dir}. Use --force to replace."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    command = training_command(clips_dir, output_dir, epochs, threads)
    config = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(clips_dir),
        "output_dir": str(output_dir),
        "class_counts": counts,
        "total_clips": total,
        "validation_split": float(VALIDATION_SPLIT),
        "learning_rate": float(LEARNING_RATE),
        "hidden_units": int(HIDDEN_UNITS),
        "dropout": float(DROPOUT),
        "max_epochs": epochs,
        "batch_size": int(BATCH_SIZE),
        "threads": threads,
        "command": command,
    }
    (output_dir / "training_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Training clip counts: {counts} (total={total})")
    print("Running training command:")
    print(" ".join(command))
    result = subprocess.run(command, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        raise RuntimeError(
            f"BirdNET training failed with return code {result.returncode}"
        )
    validate_artifacts(output_dir)
    print(f"Model written to {output_dir}")


def parse_args() -> argparse.Namespace:
    """解析双时期模型训练与单时期重跑选项。"""

    parser = argparse.ArgumentParser(
        description="Train adult and chick BirdNET classifiers from date-split clips."
    )
    parser.add_argument(
        "--epochs", type=int, default=int(os.environ.get("BIRDNET_EPOCHS", "50"))
    )
    parser.add_argument(
        "--threads", type=int, default=int(os.environ.get("BIRDNET_THREADS", "8"))
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete and replace existing model output directories.",
    )
    parser.add_argument(
        "--adult-only",
        action="store_true",
        help="Train only the adult-period model.",
    )
    parser.add_argument(
        "--chick-only",
        action="store_true",
        help="Train only the chick-period model.",
    )
    return parser.parse_args()


def main() -> None:
    """训练请求的成鸟期和/或雏鸟期模型。"""

    args = parse_args()
    configure_cache()
    print_config()

    models_dir = CHICK_SCREENING_DIR / "models"

    if not args.chick_only:
        adult_clips = CHICK_SCREENING_DIR / "adult_clips"
        adult_output = models_dir / "adult_model"
        print(f"\n=== Training adult model ===")
        print(f"Input: {adult_clips}")
        print(f"Output: {adult_output}")
        train_one_model(adult_clips, adult_output, args.epochs, args.threads, args.force)

    if not args.adult_only:
        chick_clips = CHICK_SCREENING_DIR / "chick_clips"
        chick_output = models_dir / "chick_model"
        print(f"\n=== Training chick model ===")
        print(f"Input: {chick_clips}")
        print(f"Output: {chick_output}")
        train_one_model(chick_clips, chick_output, args.epochs, args.threads, args.force)


if __name__ == "__main__":
    main()
