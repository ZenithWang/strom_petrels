"""串联日期数据准备、双模型训练、扫描和分歧分析。"""

# ============================================================
# run_pipeline.py — 双模型雏鸟筛选完整流程入口
# ============================================================
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from analyze_discrepancies import analyze_discrepancies
from dual_model_scan import dual_scan, find_recordings
from prepare_date_dataset import prepare_date_dataset, _restore_label_targets
from train_date_models import train_one_model
from project_config import (
    CHICK_SCREENING_DIR,
    print_config,
)
from data_processing.prepare_dataset import LABEL_TARGETS, index_existing_audio_files


def parse_args() -> argparse.Namespace:
    """解析完整雏鸟筛选管线的日期、训练、扫描和跳过选项。"""

    parser = argparse.ArgumentParser(
        description="Dual-model chick vocalisation screening pipeline."
    )
    # Data preparation
    parser.add_argument(
        "--adult-end", default="2021-05-31T23:59:59+00:00",
        help="Adult proxy end timestamp (ISO format).",
    )
    parser.add_argument(
        "--chick-start", default="2021-07-23T00:00:00+00:00",
        help="Chick proxy start timestamp (ISO format).",
    )
    parser.add_argument(
        "--target-positive", type=int, default=None,
        help="Target positive clips per period (scales negatives proportionally).",
    )
    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--threads", type=int, default=8)
    # Scanning
    parser.add_argument("--recordings-dir", type=Path,
                       help="Directory of chick-proxy recordings to scan.")
    parser.add_argument("--batch-size", type=int, default=32)
    # Analysis
    parser.add_argument("--adult-threshold", type=float, default=0.80)
    parser.add_argument("--chick-threshold", type=float, default=0.80)
    parser.add_argument("--representative-count", type=int, default=12)
    # Control
    parser.add_argument("--force", action="store_true",
                        help="Replace existing outputs.")
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-scan", action="store_true")
    parser.add_argument("--skip-analysis", action="store_true")
    return parser.parse_args()


def main() -> None:
    """按请求顺序运行双模型雏鸟筛选各阶段。"""

    args = parse_args()
    print_config()

    original_counts = {t["db_label"]: t["count"] for t in LABEL_TARGETS}

    # ---- Step 1: Prepare data ----
    if not args.skip_prepare:
        audio_index = index_existing_audio_files()
        print(f"Indexed {len(audio_index)} audio files")

        try:
            adult_dir = CHICK_SCREENING_DIR / "adult_clips"
            print(f"\n=== Step 1a: Adult period (<= {args.adult_end}) ===")
            prepare_date_dataset(
                adult_dir, date_end=args.adult_end,
                target_positive=args.target_positive,
                audio_index=audio_index, force=args.force,
            )
            _restore_label_targets(original_counts)

            chick_dir = CHICK_SCREENING_DIR / "chick_clips"
            print(f"\n=== Step 1b: Chick period (>= {args.chick_start}) ===")
            prepare_date_dataset(
                chick_dir, date_start=args.chick_start,
                target_positive=args.target_positive,
                audio_index=audio_index, force=args.force,
            )
        finally:
            _restore_label_targets(original_counts)

    # ---- Step 2: Train models ----
    if not args.skip_train:
        models_dir = CHICK_SCREENING_DIR / "models"

        print("\n=== Step 2a: Train adult model ===")
        train_one_model(
            CHICK_SCREENING_DIR / "adult_clips",
            models_dir / "adult_model",
            args.epochs, args.threads, args.force,
        )

        print("\n=== Step 2b: Train chick model ===")
        train_one_model(
            CHICK_SCREENING_DIR / "chick_clips",
            models_dir / "chick_model",
            args.epochs, args.threads, args.force,
        )

    # ---- Step 3: Dual scan ----
    if not args.skip_scan:
        if not args.recordings_dir:
            print("Skipping scan: --recordings-dir not provided.")
        else:
            print("\n=== Step 3: Dual-model scan ===")
            recordings = find_recordings(args.recordings_dir.expanduser().resolve())
            print(f"Found {len(recordings)} recordings")
            adult_model = CHICK_SCREENING_DIR / "models" / "adult_model" / "model.tflite"
            chick_model = CHICK_SCREENING_DIR / "models" / "chick_model" / "model.tflite"
            dual_scan(
                recordings,
                CHICK_SCREENING_DIR / "dual_scores.csv",
                adult_model.expanduser().resolve(),
                chick_model.expanduser().resolve(),
                args.batch_size, args.force,
            )

    # ---- Step 4: Analysis ----
    if not args.skip_analysis:
        scores_csv = CHICK_SCREENING_DIR / "dual_scores.csv"
        if not scores_csv.is_file():
            print(f"Skipping analysis: {scores_csv} not found. Run scan first.")
        else:
            print("\n=== Step 4: Discrepancy analysis ===")
            analyze_discrepancies(
                scores_csv,
                CHICK_SCREENING_DIR / "analysis",
                args.adult_threshold, args.chick_threshold,
                args.representative_count,
            )

    print("\n=== Pipeline complete ===")
    print(f"Outputs: {CHICK_SCREENING_DIR}")


if __name__ == "__main__":
    main()
