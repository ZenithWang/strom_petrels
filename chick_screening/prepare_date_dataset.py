"""按成鸟/雏鸟代理日期从数据库准备两个平衡训练数据集。"""

# ============================================================
# prepare_date_dataset.py
# 从 SQLite 数据库按日期范围筛选候选检测，
# 使用 MILP 平衡选取，切出 3 秒 WAV 片段，
# 生成 BirdNET 兼容的类别目录结构。
# ============================================================
from __future__ import annotations

import csv
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
DATA_PROCESSING_DIR = SCRIPTS_DIR / "data_processing"
if str(DATA_PROCESSING_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_PROCESSING_DIR))

from prepare_dataset import (
    CLIP_SECONDS,
    LABEL_TARGETS,
    MAX_CLIPS_PER_CLASS_PER_SOURCE,
    SAMPLE_RATE,
    Candidate,
    SampleSelectionError,
    _overlap_pairs,
    candidate_interval,
    extract_clip,
    find_audio_file,
    index_existing_audio_files,
    non_overlapping_candidates,
    optimize_balanced_selection,
    overlaps,
    read_candidates,
)
from project_config import (
    AUDIO_ROOT,
    CHICK_SCREENING_DIR,
    DB_PATH,
    print_config,
)

POSITIVE_FOLDER = "storm_petrel"
NEGATIVE_FOLDER = "not_storm_petrel"

DATE_FIELDS = [
    "clip_id", "label", "target_label", "negative_type",
    "species", "call_type", "source_file", "source_path",
    "clip_path", "start_seconds", "end_seconds", "score",
]


def _clip_id(candidate: Candidate, rank: int) -> str:
    source = Path(candidate.recording_file).stem
    start = int(round(candidate.start_seconds * 1000))
    prefix = "storm_petrel" if candidate.target_label == "1" else candidate.negative_type
    return f"{prefix}_{rank:05d}_{source}_{start:010d}"


def _compute_feasible_targets(
    positive_pool: dict[str, list[Candidate]],
    negative_pool: dict[str, list[Candidate]],
    target_positive: int | None,
) -> int:
    """根据每来源瓶颈分析，计算可行的正例目标数。

    关键：每来源 max_clips = min(pos_count, 4 × min_over_types(neg_type_count))
    因为四种负类比例为 1:1:1:1，正例=负例总数。
    """
    negative_targets = [t for t in LABEL_TARGETS if t["target_label"] == "0"]

    # Per-source per-negative-type non-overlapping count
    neg_by_type: dict[str, dict[str, int]] = {nt["db_label"]: {} for nt in negative_targets}
    for nt in negative_targets:
        for source in positive_pool:
            count = sum(
                1 for c in negative_pool.get(source, [])
                if c.db_label == nt["db_label"]
            )
            neg_by_type[nt["db_label"]][source] = count

    # Per-source bottleneck
    source_max = {}
    total_max = 0
    for source, pos_rows in positive_pool.items():
        pos_count = min(len(pos_rows), MAX_CLIPS_PER_CLASS_PER_SOURCE)
        neg_min = pos_count  # default: limited by positive
        for nt in negative_targets:
            neg_count = neg_by_type[nt["db_label"]].get(source, 0)
            if neg_count < neg_min:
                neg_min = neg_count
        # Each neg type contributes 1/4 of total; total = 4 * min(neg)
        source_clips = min(pos_count, 4 * neg_min)
        # But also can't exceed MAX_CLIPS_PER_CLASS_PER_SOURCE total
        source_clips = min(source_clips, MAX_CLIPS_PER_CLASS_PER_SOURCE)
        source_max[source] = source_clips
        total_max += source_clips

    if not source_max:
        raise SampleSelectionError("No sources with positive candidates found.")

    # Check which negative types have zero across all active sources
    for nt in negative_targets:
        total_neg = sum(neg_by_type[nt["db_label"]].values())
        if total_neg == 0:
            raise SampleSelectionError(
                f"No {nt['db_label']} candidates found in any source "
                f"that also has positive candidates."
            )

    # Print per-source detail
    sources = sorted(source_max)
    for source in sources:
        pos = min(len(positive_pool[source]), MAX_CLIPS_PER_CLASS_PER_SOURCE)
        negs = {nt["db_label"]: neg_by_type[nt["db_label"]].get(source, 0) for nt in negative_targets}
        bottleneck = min(negs, key=negs.get) if negs else ""
        if source_max[source] < pos:
            print(
                f"  {Path(source).name}: pos={pos}, "
                f"bottleneck={bottleneck}={negs[bottleneck]}, "
                f"max_clips={source_max[source]}"
            )
        else:
            print(f"  {Path(source).name}: pos={pos}, max_clips={source_max[source]}")

    # 原始默认目标
    if target_positive is None:
        target_positive = sum(t["count"] for t in LABEL_TARGETS if t["target_label"] == "1")

    if target_positive > total_max:
        feasible = max(1, int(total_max))
        print(
            f"Target positive ({target_positive}) exceeds feasible max ({total_max}) "
            f"with {len(sources)} sources. Auto-scaling to {feasible}."
        )
        target_positive = feasible
    else:
        feasible = target_positive

    _adjust_label_targets(feasible)

    print(
        f"Source files: {len(sources)}, "
        f"target positive: {feasible}"
    )
    for nt in negative_targets:
        print(f"  {nt['db_label']}: target={nt['count']}")

    return feasible


def select_date_samples(
    date_end: str | None = None,
    date_start: str | None = None,
    target_positive: int | None = None,
    audio_index: dict[str, Path] | None = None,
) -> list[Candidate]:
    """按日期范围从 DB 选取候选，运行 MILP 优化选择。"""
    # Step 1: 从 DB 读取日期过滤后的候选
    candidates_by_target: dict[str, list[Candidate]] = {}
    for target in LABEL_TARGETS:
        candidates = read_candidates(
            target["db_label"],
            target["label"],
            target["target_label"],
            target["negative_type"],
            date_end=date_end,
            date_start=date_start,
        )
        if audio_index is not None:
            candidates = [
                c for c in candidates
                if find_audio_file(c.recording_file, audio_index) is not None
            ]
        candidates_by_target[target["db_label"]] = candidates

    # Step 2: 正例候选 -> 按来源分组 -> 去重叠
    positive_target = next(t for t in LABEL_TARGETS if t["target_label"] == "1")
    raw_positive_by_source: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates_by_target[positive_target["db_label"]]:
        raw_positive_by_source[candidate.recording_file].append(candidate)

    positive_pool = {
        source: non_overlapping_candidates(rows, MAX_CLIPS_PER_CLASS_PER_SOURCE)
        for source, rows in raw_positive_by_source.items()
    }
    positive_intervals = {
        source: [candidate_interval(c) for c in rows]
        for source, rows in positive_pool.items()
    }

    # Step 3: 负例候选 -> 过滤与正例重叠的 -> 去重叠
    negative_pool: dict[str, list[Candidate]] = defaultdict(list)
    negative_targets = [t for t in LABEL_TARGETS if t["target_label"] == "0"]
    for target in negative_targets:
        by_source: dict[str, list[Candidate]] = defaultdict(list)
        for candidate in candidates_by_target[target["db_label"]]:
            if candidate.recording_file not in positive_pool:
                continue
            start, end = candidate_interval(candidate)
            if overlaps(start, end, positive_intervals[candidate.recording_file]):
                continue
            by_source[candidate.recording_file].append(candidate)
        for source, rows in by_source.items():
            negative_pool[source].extend(
                non_overlapping_candidates(rows, MAX_CLIPS_PER_CLASS_PER_SOURCE)
            )

    # Step 4: 自动计算可行目标数并调整 LABEL_TARGETS
    feasible_target = _compute_feasible_targets(positive_pool, negative_pool, target_positive)

    # Step 5: MILP 优化（如不可行则逐步降目标重试）
    selected = _try_milp_with_fallback(positive_pool, negative_pool, feasible_target)
    return selected


def _try_milp_with_fallback(
    positive_pool: dict[str, list[Candidate]],
    negative_pool: dict[str, list[Candidate]],
    target_positive: int,
    max_retries: int = 5,
) -> list[Candidate]:
    """运行 MILP，如果不可行则逐步降低 target 重试。"""
    current_target = target_positive
    for attempt in range(max_retries):
        _adjust_label_targets(current_target)
        try:
            return optimize_balanced_selection(positive_pool, negative_pool)
        except SampleSelectionError:
            if attempt == max_retries - 1:
                raise
            reduced = max(1, int(current_target * 0.75))
            print(
                f"MILP infeasible at target={current_target}, "
                f"retrying at target={reduced} (attempt {attempt + 2}/{max_retries})"
            )
            current_target = reduced
    # unreachable
    raise SampleSelectionError("MILP infeasible after all retries.")


def _adjust_label_targets(target_positive: int) -> None:
    """按比例调整各标签的目标数量，保证正例=负例总和。"""
    negative_targets = [t for t in LABEL_TARGETS if t["target_label"] == "0"]
    original_negative_total = sum(t["count"] for t in negative_targets)

    # 先按比例缩放各负类，向下取整
    for nt in negative_targets:
        nt["count"] = int(target_positive * nt["count"] / original_negative_total)

    # 因为向下取整可能有剩余，分配给当前 count 最大的负类
    remaining = target_positive - sum(nt["count"] for nt in negative_targets)
    if remaining > 0:
        negative_targets.sort(key=lambda t: t["count"], reverse=True)
        for i in range(remaining):
            negative_targets[i % len(negative_targets)]["count"] += 1

    # 正类目标
    for t in LABEL_TARGETS:
        if t["target_label"] == "1":
            t["count"] = target_positive


def prepare_date_dataset(
    output_dir: Path,
    date_end: str | None = None,
    date_start: str | None = None,
    target_positive: int | None = None,
    audio_index: dict[str, Path] | None = None,
    force: bool = False,
) -> int:
    """完整的日期过滤数据准备流程。返回选中 clip 数量。"""
    if output_dir.exists():
        if not force:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. Use --force to replace."
            )
        shutil.rmtree(output_dir)

    selected = select_date_samples(
        date_end=date_end,
        date_start=date_start,
        target_positive=target_positive,
        audio_index=audio_index,
    )

    if not selected:
        raise ValueError("No candidates were selected for the given date range.")

    # 按来源解析音频文件
    sources: dict[str, Path] = {}
    if audio_index is not None:
        sources = {
            rf: find_audio_file(rf, audio_index)
            for rf in sorted({c.recording_file for c in selected})
        }
        missing = [rf for rf, p in sources.items() if p is None]
        if missing:
            raise FileNotFoundError(f"Missing source recordings: {missing}")

    # 切出 clips 并写入标签
    rows: list[dict[str, str]] = []
    positive_dir = output_dir / POSITIVE_FOLDER
    negative_dir = output_dir / NEGATIVE_FOLDER
    positive_dir.mkdir(parents=True, exist_ok=True)
    negative_dir.mkdir(parents=True, exist_ok=True)

    for rank, candidate in enumerate(selected, start=1):
        identifier = _clip_id(candidate, rank)
        class_folder = POSITIVE_FOLDER if candidate.target_label == "1" else NEGATIVE_FOLDER
        destination = output_dir / class_folder / f"{identifier}.wav"
        source_path = sources.get(candidate.recording_file, Path(candidate.recording_file))
        extract_clip(source_path, destination, candidate.start_seconds)
        rows.append({
            "clip_id": identifier,
            "label": candidate.label,
            "target_label": candidate.target_label,
            "negative_type": candidate.negative_type,
            "species": candidate.species,
            "call_type": "",
            "source_file": candidate.recording_file,
            "source_path": str(source_path),
            "clip_path": str(destination),
            "start_seconds": f"{candidate.start_seconds:.3f}",
            "end_seconds": f"{candidate.start_seconds + CLIP_SECONDS:.3f}",
            "score": f"{candidate.score:.6f}",
        })

    # 写标签 CSV
    labels_path = output_dir / "labels.csv"
    with labels_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DATE_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    # 打印摘要
    target_counts = Counter(r["target_label"] for r in rows)
    negative_counts = Counter(r["negative_type"] for r in rows if r["target_label"] == "0")
    source_count = len({r["source_file"] for r in rows})
    print(f"Wrote {len(rows)} clips to {output_dir}")
    print(f"  Positive: {target_counts['1']}; Negative: {target_counts['0']}")
    print(f"  Negative types: {dict(sorted(negative_counts.items()))}")
    print(f"  Source files: {source_count}")
    print(f"  Labels CSV: {labels_path}")
    return len(rows)


def _restore_label_targets(original_counts: dict[str, int]) -> None:
    """恢复 LABEL_TARGETS 中的原始 count 值。"""
    for target in LABEL_TARGETS:
        if target["db_label"] in original_counts:
            target["count"] = original_counts[target["db_label"]]


def main() -> None:
    """解析日期与目标数量，并准备请求的一个或两个时期数据集。"""

    import argparse
    from datetime import date

    parser = argparse.ArgumentParser(
        description="Prepare date-filtered training datasets for adult/chick models."
    )
    parser.add_argument(
        "--adult-end",
        default="2021-05-31T23:59:59+00:00",
        help="Adult proxy end timestamp (ISO format).",
    )
    parser.add_argument(
        "--chick-start",
        default="2021-07-23T00:00:00+00:00",
        help="Chick proxy start timestamp (ISO format).",
    )
    parser.add_argument(
        "--target-positive",
        type=int,
        default=None,
        help="Target number of positive clips per period (scales negatives proportionally).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Remove and replace existing output directories.",
    )
    parser.add_argument(
        "--adult-only",
        action="store_true",
        help="Prepare only the adult-period dataset.",
    )
    parser.add_argument(
        "--chick-only",
        action="store_true",
        help="Prepare only the chick-period dataset.",
    )
    args = parser.parse_args()

    print_config()
    print(f"Database: {DB_PATH}")

    # 保存原始 count 值以便恢复
    original_counts = {t["db_label"]: t["count"] for t in LABEL_TARGETS}

    audio_index = index_existing_audio_files()
    print(f"Audio root: {AUDIO_ROOT} ({len(audio_index)} files indexed)")

    try:
        if not args.chick_only:
            adult_dir = CHICK_SCREENING_DIR / "adult_clips"
            print(f"\n=== Adult period (timestamp <= {args.adult_end}) ===")
            count = prepare_date_dataset(
                adult_dir,
                date_end=args.adult_end,
                target_positive=args.target_positive,
                audio_index=audio_index,
                force=args.force,
            )
            print(f"Adult dataset: {count} clips")
            _restore_label_targets(original_counts)

        if not args.adult_only:
            chick_dir = CHICK_SCREENING_DIR / "chick_clips"
            print(f"\n=== Chick period (timestamp >= {args.chick_start}) ===")
            count = prepare_date_dataset(
                chick_dir,
                date_start=args.chick_start,
                target_positive=args.target_positive,
                audio_index=audio_index,
                force=args.force,
            )
            print(f"Chick dataset: {count} clips")
    finally:
        _restore_label_targets(original_counts)

    print("\n=== Next steps ===")
    print("python chick_screening/train_date_models.py --epochs 50 --threads 8")


if __name__ == "__main__":
    main()
