from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
SCRIPTS_DIR = SCRIPT_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import build_birdnet_fold_dataset
import prepare_dataset
from project_config import PROJECT_ROOT, print_config

TRAINING_SCRIPT = PROJECT_ROOT / "scripts" / "model_training" / "run_nested_cv.py"
ANALYSIS_SCRIPT = PROJECT_ROOT / "scripts" / "results_analysis" / "summarize_nested_cv.py"


def run_step(name: str, action) -> None:
    print(f"\n=== {name} ===")
    started = perf_counter()
    action()
    elapsed = perf_counter() - started
    print(f"Finished {name} in {elapsed:.1f}s")


def main() -> None:
    print_config()
    run_step("按指定标签准备二分类数据和 stratified grouped nested CV folds", prepare_dataset.main)
    run_step("生成 nested CV BirdNET 数据集目录", build_birdnet_fold_dataset.main)

    print("\n=== 下一步 ===")
    print(f"训练脚本: {TRAINING_SCRIPT}")
    print(f"结果汇总脚本: {ANALYSIS_SCRIPT}")


if __name__ == "__main__":
    main()
