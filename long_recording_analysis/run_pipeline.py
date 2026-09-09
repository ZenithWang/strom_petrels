"""串联长录音推理与候选截取的便捷命令行入口。"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from extract_candidates import DEFAULT_THRESHOLD, extract_candidates
from project_config import DEPLOYMENT_MODEL_DIR, LONG_RECORDING_ANALYSIS_DIR, print_config
from scan_long_recordings import DEFAULT_STEP_SECONDS, scan_recordings


def parse_args() -> argparse.Namespace:
    """解析长录音完整运行的模型、步长、阈值和输出选项。"""

    parser = argparse.ArgumentParser(description="Scan non-overlapping three-second windows and extract candidates.")
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--run-name", help="Output subdirectory name; defaults to a local timestamp.")
    parser.add_argument("--model", type=Path, default=DEPLOYMENT_MODEL_DIR / "model.tflite")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--step", type=float, default=DEFAULT_STEP_SECONDS)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--force", action="store_true", help="Replace all outputs for this run name.")
    return parser.parse_args()


def validate_run_name(value: str) -> str:
    """验证输出子目录名，阻止路径穿越和无效名称。"""

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value):
        raise ValueError("run_name may contain only letters, digits, dots, underscores, and hyphens")
    return value


def main() -> None:
    """依次运行窗口评分和候选音频截取。"""

    args = parse_args()
    run_name = validate_run_name(args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S"))
    run_dir = LONG_RECORDING_ANALYSIS_DIR / run_name
    all_windows_csv = run_dir / "all_windows.csv"
    detections_csv = run_dir / "detections.csv"
    candidates_dir = run_dir / "candidates"
    print_config()
    print(f"Run directory: {run_dir}")
    scan_recordings(
        args.input_dir.expanduser().resolve(),
        all_windows_csv,
        args.model.expanduser().resolve(),
        args.step,
        args.batch_size,
        args.force,
    )
    extract_candidates(
        all_windows_csv,
        detections_csv,
        candidates_dir,
        args.threshold,
        args.force,
    )


if __name__ == "__main__":
    main()
