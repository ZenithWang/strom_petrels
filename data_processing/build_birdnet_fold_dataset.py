"""把 fold 清单物化为 BirdNET-Analyzer 要求的类别目录结构。"""

from __future__ import annotations

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import csv
import os
import shutil
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from project_config import CLIPS_DIR, DATASET_ROOT, FOLDS_PATH, print_config

OUTPUT_ROOT = DATASET_ROOT
POSITIVE_FOLDER = "storm_petrel"
NEGATIVE_FOLDER = "not_storm_petrel"

def read_rows(path: Path) -> list[dict[str, str]]:
    """读取 nested CV fold 表。"""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [row for row in csv.DictReader(handle) if row.get("clip_id")]


def class_folder(row: dict[str, str]) -> str:
    """BirdNET 通过文件夹名识别类别；二分类负类统一放入 not_storm_petrel。"""
    return POSITIVE_FOLDER if row["target_label"] == "1" else NEGATIVE_FOLDER


def copy_rows(rows: list[dict[str, str]], destination_root: Path) -> Counter[str]:
    """用硬链接把 clips 放到 BirdNET 文件夹结构。"""
    counts: Counter[str] = Counter()
    for row in rows:
        folder = class_folder(row)
        source = CLIPS_DIR / folder / f"{row['clip_id']}.wav"
        target_dir = destination_root / folder
        target_dir.mkdir(parents=True, exist_ok=True)
        os.link(source, target_dir / source.name)
        counts[folder] += 1
    for folder in (POSITIVE_FOLDER, NEGATIVE_FOLDER):
        (destination_root / folder).mkdir(parents=True, exist_ok=True)
    return counts


def build_all_fold_datasets(folds_path: Path = FOLDS_PATH, output_root: Path = OUTPUT_ROOT) -> dict[str, Counter[str]]:
    """为所有外层和内层 folds 创建 BirdNET 数据集目录。"""
    rows = read_rows(folds_path)
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    if not rows:
        return {}

    results: dict[str, Counter[str]] = {}
    for outer_fold in sorted({row["outer_fold"] for row in rows}):
        outer_rows = [row for row in rows if row["outer_fold"] == outer_fold]
        outer_train = [row for row in outer_rows if row["split"] == "outer_train"]
        outer_test = [row for row in outer_rows if row["split"] == "outer_test"]
        outer_dir = output_root / f"outer_{outer_fold}"

        results[f"outer_{outer_fold}:final_train"] = copy_rows(outer_train, outer_dir / "final_train")
        results[f"outer_{outer_fold}:test"] = copy_rows(outer_test, outer_dir / "test")

        for inner_fold in sorted({row["inner_fold"] for row in outer_train if row.get("inner_fold")}):
            inner_train = [row for row in outer_train if row["inner_fold"] != inner_fold]
            inner_validation = [row for row in outer_train if row["inner_fold"] == inner_fold]
            inner_dir = outer_dir / f"inner_{inner_fold}"
            results[f"outer_{outer_fold}:inner_{inner_fold}:train"] = copy_rows(inner_train, inner_dir / "train")
            results[f"outer_{outer_fold}:inner_{inner_fold}:validation"] = copy_rows(inner_validation, inner_dir / "validation")
    return results


def main() -> None:
    """构建全部 outer/inner fold 数据集并打印计数。"""

    print_config()
    results = build_all_fold_datasets()
    print(f"已写出 nested CV BirdNET 数据集目录 {OUTPUT_ROOT}")
    for name, counts in sorted(results.items()):
        print(f"{name}: storm_petrel={counts[POSITIVE_FOLDER]}, not_storm_petrel={counts[NEGATIVE_FOLDER]}")


if __name__ == "__main__":
    main()
