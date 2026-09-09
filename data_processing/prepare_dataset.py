"""从检测数据库构建平衡三秒 clips，并生成 grouped nested-CV fold 方案。"""

# ============================================================
# 自动编码：prepare_dataset.py
# 主要作用：从 SQLite 数据库读取鸟鸣检测结果，
# 筛选 balanced (正负各半)、non-overlapping 的 3 秒音频片段，
# 用 MILP（混合整数线性规划）做最优化选择，
# 最后切出片段、写入标签 CSV、并分配 grouped nested CV folds。
# ============================================================
from __future__ import annotations

import csv
import shutil
import sqlite3
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from math import ceil, floor
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

# 强制 stdout 使用 UTF-8 编码，防止输出非 ASCII 字符时报错
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# 将项目根目录加入 sys.path，以便从 project_config 模块导入常量
SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from project_config import (
    AUDIO_ROOT,          # 原始音频文件根目录
    BINARY_LABELS_PATH,  # 二分类标签 CSV 输出路径
    CLIPS_DIR,           # 切好的 3 秒 WAV 片段输出目录
    DB_PATH,             # SQLite 检测数据库路径
    DOWNLOAD_FILES_PATH, # 下载清单 CSV 输出路径
    FOLD_BALANCE_REPORT_PATH,  # fold 平衡报告 CSV 输出路径
    FOLDS_PATH,          # fold 分配 CSV 输出路径
    print_config,        # 打印配置信息的工具函数
)

# ============================================================
# 全局常量
# ============================================================
SAMPLE_RATE = 48000          # 音频采样率：48 kHz
CLIP_SECONDS = 3.0            # 每个片段时长：3 秒

POSITIVE_FOLDER = "storm_petrel"    # 正例音频片段输出目录名
NEGATIVE_FOLDER = "not_storm_petrel" # 负例音频片段输出目录名

OUTER_FOLDS = 5                 # 外层交叉验证折数（5-fold）
INNER_FOLDS = 3                 # 内层交叉验证折数（3-fold）
FOLD_SIZE_TOLERANCE = 0.10      # 各 fold 之间的大小允许偏差 ±10%

# 每个来源录音文件最多从每个类别选多少个候选片段
# 3 sources * 73 positive/negative pairs * 2 classes = 438 rows，正好塞进 440 行以内的 outer test 折
MAX_CLIPS_PER_CLASS_PER_SOURCE = 73

# MILP 求解器最多跑 300 秒
SELECTION_TIME_LIMIT_SECONDS = 300

# ============================================================
# LABEL_TARGETS: 数据集的类别定义
# 每条记录：
#   db_label     - 数据库中的检测标签
#   count        - 该数据库标签的目标样本数（负例子类型按此汇总，当前各500）
#   label        - 二分类标签文本
#   target_label - 数字标签（1=storm petrel, 0=not）
#   negative_type - 负例的来源类型（other_bird / environment_noise）
# ============================================================
LABEL_TARGETS = (
    {
        "db_label": "Hydrobates pelagicus_European Storm-Petrel",
        "count": 1000,
        "label": "storm_petrel",
        "target_label": "1",
        "negative_type": "",
    },
    {
        "db_label": "Puffinus puffinus_Manx Shearwater",
        "count": 250,
        "label": "not_storm_petrel",
        "target_label": "0",
        "negative_type": "other_bird",
    },
    {
        "db_label": "Larus argentatus_Herring Gull",
        "count": 250,
        "label": "not_storm_petrel",
        "target_label": "0",
        "negative_type": "other_bird",
    },
    {
        "db_label": "Engine_Engine",
        "count": 250,
        "label": "not_storm_petrel",
        "target_label": "0",
        "negative_type": "environment_noise",
    },
    {
        "db_label": "Dog_Dog",
        "count": 250,
        "label": "not_storm_petrel",
        "target_label": "0",
        "negative_type": "environment_noise",
    },
)

# ============================================================
# CSV 字段定义
# ============================================================
# 二分类标签 CSV 的列：最终输出给 ML 训练的完整标签表
BINARY_FIELDS = [
    "clip_id",       # 片段唯一标识
    "label",         # 文本标签（storm_petrel / not_storm_petrel）
    "target_label",  # 数字标签（1 或 0）
    "negative_type", # 负例类型（other_bird / environment_noise，正例为空）
    "species",       # 检测到的物种名
    "call_type",     # 叫声类型（保留字段，当前为空）
    "source_file",   # 来源录音文件名
    "source_path",   # 来源录音文件的完整路径
    "clip_path",     # 切出来的 3 秒片段文件路径
    "start_seconds", # 片段在原录音中的起始秒数
    "end_seconds",   # 片段结束秒数（=start + 3）
    "score",         # 检测置信度
    "split",         # 划分集（outer_train / outer_test）
    "outer_fold",    # 外层折号（01~05）
    "inner_fold",    # 内层折号（01~03，outer_test 行为空）
]

# 下载清单 CSV 的列：给人工/脚本下载用的
DOWNLOAD_FIELDS = [
    "download_order",   # 下载顺序
    "recording_file",   # 录音文件
    "location",         # 录制地点编号
    "timestamp",        # 录制时间戳
    "selected_clips",   # 从此录音中选取的片段数
    "labels",           # 此录音包含的标签类别
    "best_score",       # 此录音中最好的一次检测分数
]

# 平衡报告 CSV 的列：每个 fold split 的详细统计数据
BALANCE_REPORT_FIELDS = [
    "outer_fold",        # 外层折号
    "inner_fold",        # 内层折号
    "split",             # 划分集（train / test / validation）
    "total",             # 总样本数
    "positive",          # 正例数
    "negative",          # 负例数
    "positive_ratio",    # 正例占比（理想应为 0.5）
    "environment_noise", # 环境噪音负例数
    "other_bird",        # 其他鸟鸣负例数
    "source_files",      # 来源录音文件数
]


# ============================================================
# 数据类：
#   Candidate  - 一个候选检测片段，包含从 DB 读取的原始信息
#   FoldPlan   - 外层和内层的 fold 分配方案
# ============================================================
@dataclass(frozen=True)
class Candidate:
    """保存一个数据库检测候选及其二分类元数据。"""

    db_label: str       # 数据库中的物种/声音标签
    species: str        # 物种名
    score: float        # 检测置信度
    location: int       # 录制地点编号
    timestamp: str      # 录制时间戳
    recording_file: str # 录音文件名
    duration_seconds: float  # 录音总时长
    start_seconds: float     # 检测开始时间
    end_seconds: float       # 检测结束时间
    label: str               # 二分类文本标签
    target_label: str        # 数字标签
    negative_type: str       # 负例类型


@dataclass(frozen=True)
class FoldPlan:
    """保存 outer 与 inner 层按来源录音分组的 fold 分配。"""

    outer_assignments: dict[str, int]          # source_file -> outer fold number
    inner_assignments: dict[int, dict[str, int]] # outer_fold -> {source_file -> inner fold}


# ============================================================
# 自定义异常
# ============================================================
class FoldAssignmentError(ValueError):
    """当 grouped folds 无法满足平衡约束时抛出。"""


class SampleSelectionError(ValueError):
    """当 balanced、non-overlapping 的样本选择不可行时抛出。"""


# ============================================================
# 第 1 阶段：候选检测读取 (第164-223行)
# ============================================================

def parse_detection_desc(desc: str) -> tuple[str, float]:
    """
    从检测描述字符串中解析物种名和置信度分数。
    格式如 "Hydrobates pelagicus 0.95"，最后一个空格分隔分数与物种名。
    """
    species, raw_score = desc.rsplit(" ", 1)
    return species.strip(), float(raw_score)


def read_candidates(
    db_label: str,
    label: str,
    target_label: str,
    negative_type: str,
    date_end: str | None = None,
    date_start: str | None = None,
) -> list[Candidate]:
    """
    从 SQLite 数据库中读取指定标签的检测候选 (第172行以下)。
    步骤：
      1. 检查 DB_PATH 是否存在
      2. 执行 JOIN 查询：calls 表 JOIN recordings 表获取录音文件信息
      3. 解析每个检测的 desc 字段（"物种名 分数"）
      4. 按分数降序、location、timestamp、start_seconds 升序排列

    date_end:   可选，ISO 格式时间字符串。只保留 r.timestamp <= date_end 的记录。
    date_start: 可选，ISO 格式时间字符串。只保留 r.timestamp >= date_start 的记录。
    """
    if not DB_PATH.is_file():
        raise FileNotFoundError(f"database not found: {DB_PATH}")

    date_clauses: list[str] = []
    params: list[str] = []
    if date_end is not None:
        date_clauses.append("r.timestamp <= ?")
        params.append(date_end)
    if date_start is not None:
        date_clauses.append("r.timestamp >= ?")
        params.append(date_start)

    query = """
        select
            c.location,
            c.timestamp,
            c.start,
            c.end,
            c.desc,
            r.duration,
            r.path
        from calls c
        join recordings r
          on r.location = c.location
         and r.timestamp = c.timestamp
        where c.desc like ?
    """
    for clause in date_clauses:
        query += f"          and {clause}\n"

    rows: list[Candidate] = []
    with sqlite3.connect(DB_PATH) as connection:
        for row in connection.execute(query, (f"{db_label} %", *params)):
            location, timestamp, start, end, desc, duration, recording_file = row
            species, score = parse_detection_desc(str(desc))
            rows.append(
                Candidate(
                    db_label=db_label,
                    species=species,
                    score=score,
                    location=int(location),
                    timestamp=str(timestamp),
                    recording_file=str(recording_file),
                    duration_seconds=float(duration),
                    start_seconds=float(start),
                    end_seconds=float(end),
                    label=label,
                    target_label=target_label,
                    negative_type=negative_type,
                )
            )
    # 按分数降序（高置信度优先），再按时间顺序排序
    rows.sort(
        key=lambda item: (
            -item.score,
            item.location,
            item.timestamp,
            item.start_seconds,
            item.recording_file,
        )
    )
    return rows


def overlaps(start: float, end: float, intervals: list[tuple[float, float]]) -> bool:
    """判断 [start, end) 区间是否与已选区间列表中的任一区间重叠。"""
    return any(start < interval_end and end > interval_start for interval_start, interval_end in intervals)


def index_existing_audio_files() -> dict[str, Path]:
    """
    扫描 AUDIO_ROOT 下所有文件，建立一个 名称 -> 路径 的索引。
    同时用 path.name（含扩展名）和 path.stem（不含扩展名）做键，方便查找。
    """
    if not AUDIO_ROOT.is_dir():
        raise FileNotFoundError(f"audio root not found: {AUDIO_ROOT}")

    indexed: dict[str, Path] = {}
    for path in sorted(item for item in AUDIO_ROOT.rglob("*") if item.is_file()):
        indexed.setdefault(path.name, path)
        indexed.setdefault(path.stem, path)
    return indexed


def find_audio_file(recording_file: str, audio_index: dict[str, Path]) -> Path | None:
    """
    在音频索引中查找录音文件。
    先用完整文件名（含扩展名）查找，找不到则用文件主名（不含扩展名）查找。
    """
    return audio_index.get(Path(recording_file).name) or audio_index.get(Path(recording_file).stem)


def candidate_interval(candidate: Candidate) -> tuple[float, float]:
    """获取候选片段在原录音中的 3 秒时间段。"""
    return candidate.start_seconds, candidate.start_seconds + CLIP_SECONDS


def non_overlapping_candidates(candidates: list[Candidate], limit: int) -> list[Candidate]:
    """
    从高置信度到低置信度依次选取不重叠的候选片段。
    - 按分数降序排列的 candidates 中依次尝试
    - 如果当前候选片段的时间区间与已选区间重叠，跳过
    - 达到 limit 或遍历完毕即停止
    """
    selected: list[Candidate] = []
    intervals: list[tuple[float, float]] = []
    for candidate in candidates:
        start, end = candidate_interval(candidate)
        if overlaps(start, end, intervals):
            continue
        selected.append(candidate)
        intervals.append((start, end))
        if len(selected) == limit:
            break
    return selected


def _overlap_pairs(candidates: list[Candidate]) -> list[tuple[int, int]]:
    """
    找出同一录音文件中互相重叠的候选片段索引对。
    排序后只检查时间上有可能重叠的相邻候选（剪枝优化）。
    用于后续 MILP 的互斥约束（同一 source 内的重叠负例不能同时选中）。
    """
    indexed = sorted(
        enumerate(candidates),
        key=lambda item: (item[1].start_seconds, item[1].start_seconds + CLIP_SECONDS, item[0]),
    )
    pairs: list[tuple[int, int]] = []
    for position, (left_index, left) in enumerate(indexed):
        left_end = left.start_seconds + CLIP_SECONDS
        for right_index, right in indexed[position + 1 :]:
            if right.start_seconds >= left_end:
                break  # 由于已排序，后续的 start 只会更大，不可能重叠
            pairs.append((left_index, right_index))
    return pairs


# ============================================================
# 第 2 阶段：MILP 平衡最优化选择 (第237-337行)
# ============================================================

def optimize_balanced_selection(
    positive_pool: dict[str, list[Candidate]],
    negative_pool: dict[str, list[Candidate]],
) -> list[Candidate]:
    """
    使用混合整数线性规划 (MILP) 选择精确平衡的样本集。

    约束条件（按优先级）：
      C1: 正例总数达到 LABEL_TARGETS 中配置的精确目标
      C2: 每个负例数据库标签达到 LABEL_TARGETS 中配置的精确目标
          （当前两个 negative_type 汇总后各 500）
      C3: 每个来源录音文件中，正例数 = 负例数（平衡配对）
      C4: 每个来源每个类别的片段数 <= MAX_CLIPS_PER_CLASS_PER_SOURCE
      C5: 同一来源内重叠的负例片段不能同时被选中（互斥约束）

    优化目标（三级顺序优化）：
      Stage 1: 最大化正例的检测分数之和
      Stage 2: 最大化负例的检测分数之和（在 Stage 1 最优解附近锁定）
      Stage 3: 用索引序号作为微调 tie-breaker（确定性）
    """
    # ---- 构建候选列表与索引 ----
    candidates: list[Candidate] = []
    indices_by_source: dict[str, list[int]] = defaultdict(list)
    positive_indices: list[int] = []
    negative_indices_by_label: dict[str, list[int]] = defaultdict(list)
    local_negative_indices: dict[str, list[int]] = defaultdict(list)

    for source in sorted(set(positive_pool) | set(negative_pool)):
        for candidate in positive_pool.get(source, []):
            index = len(candidates)
            candidates.append(candidate)
            indices_by_source[source].append(index)
            positive_indices.append(index)
        for candidate in negative_pool.get(source, []):
            index = len(candidates)
            candidates.append(candidate)
            indices_by_source[source].append(index)
            negative_indices_by_label[candidate.db_label].append(index)
            local_negative_indices[source].append(index)

    if not candidates:
        raise SampleSelectionError("No usable candidates were found.")

    # ---- 构建约束矩阵 ----
    row_indices: list[int] = []
    column_indices: list[int] = []
    values: list[float] = []
    lower_bounds: list[float] = []
    upper_bounds: list[float] = []

    def add_constraint(coefficients: dict[int, float], lower: float, upper: float) -> None:
        """向稀疏约束矩阵添加一行约束：lower <= sum(coefficients * x) <= upper"""
        row = len(lower_bounds)
        for column, value in coefficients.items():
            row_indices.append(row)
            column_indices.append(column)
            values.append(value)
        lower_bounds.append(lower)
        upper_bounds.append(upper)

    # C1: 正例总数精确等于目标值
    positive_target = next(target["count"] for target in LABEL_TARGETS if target["target_label"] == "1")
    add_constraint({index: 1.0 for index in positive_indices}, positive_target, positive_target)

    # C2: 各负例数据库标签数量精确等于目标值
    for target in LABEL_TARGETS:
        if target["target_label"] == "0":
            indices = negative_indices_by_label[target["db_label"]]
            add_constraint({index: 1.0 for index in indices}, target["count"], target["count"])

    # C3: 每个来源中，正例数 = 负例数（正例系数=+1，负例系数=-1，和=0）
    for source, indices in indices_by_source.items():
        coefficients = {
            index: 1.0 if candidates[index].target_label == "1" else -1.0
            for index in indices
        }
        add_constraint(coefficients, 0.0, 0.0)

        # C4: 每个来源正例数不超过上限
        positive_source_indices = [index for index in indices if candidates[index].target_label == "1"]
        add_constraint(
            {index: 1.0 for index in positive_source_indices},
            0.0,
            MAX_CLIPS_PER_CLASS_PER_SOURCE,
        )

    # C5: 同一来源内重叠的负例互斥（一次最多选一个）
    for source, global_indices in local_negative_indices.items():
        local_candidates = [candidates[index] for index in global_indices]
        for left, right in _overlap_pairs(local_candidates):
            add_constraint({global_indices[left]: 1.0, global_indices[right]: 1.0}, 0.0, 1.0)

    # ---- 构建 scipy 稀疏约束矩阵 ----
    constraint_matrix = coo_matrix(
        (values, (row_indices, column_indices)),
        shape=(len(lower_bounds), len(candidates)),
    ).tocsr()
    constraints = LinearConstraint(
        constraint_matrix,
        np.asarray(lower_bounds),
        np.asarray(upper_bounds),
    )
    bounds = Bounds(np.zeros(len(candidates)), np.ones(len(candidates)))
    integrality = np.ones(len(candidates), dtype=int)  # 所有变量都是 0/1 整数

    # ---- 三级优化目标 ----
    objectives = []
    # Stage 1: 最大化正例分数的总和（用负数形式传给 MILP 做最小化）
    positive_objective = np.zeros(len(candidates))
    positive_objective[positive_indices] = [-candidates[index].score for index in positive_indices]
    objectives.append(positive_objective)

    # Stage 2: 最大化负例分数的总和
    negative_objective = np.zeros(len(candidates))
    negative_indices = [index for indices in negative_indices_by_label.values() for index in indices]
    negative_objective[negative_indices] = [-candidates[index].score for index in negative_indices]
    objectives.append(negative_objective)

    # Stage 3: 微调 tie-breaker——按索引序号排序（确保结果是确定性的）
    tie_breaker = np.arange(1, len(candidates) + 1, dtype=float)
    objectives.append(tie_breaker)

    # ---- 顺序优化（Lexicographic MILP）----
    result = None
    locked_rows: list[np.ndarray] = []   # 已锁定阶段的优化目标
    locked_upper: list[float] = []       # 各阶段最优值的上限
    for stage, objective in enumerate(objectives):
        if locked_rows:
            # 添加已锁定阶段的约束：优化值必须 <= 上一阶段最优值 + 小容忍度
            locked_matrix = coo_matrix(np.vstack(locked_rows)).tocsr()
            active_constraints = (
                constraints,
                LinearConstraint(locked_matrix, -np.inf, np.asarray(locked_upper)),
            )
        else:
            active_constraints = constraints
        result = milp(
            c=objective,
            integrality=integrality,
            bounds=bounds,
            constraints=active_constraints,
            options={"time_limit": SELECTION_TIME_LIMIT_SECONDS, "mip_rel_gap": 0.0},
        )
        if result.status != 0 or not result.success or result.x is None:
            raise SampleSelectionError(
                "Balanced sample selection did not reach a proven optimum "
                f"at stage {stage + 1}: status={result.status}, {result.message}"
            )
        if stage < len(objectives) - 1:
            # 记录本阶段最优值，作为下一阶段的约束
            locked_rows.append(objective)
            locked_upper.append(float(result.fun) + 1e-7)

    # ---- 提取被选中的候选（x > 0.5 即为选中）----
    assert result is not None and result.x is not None
    selected = [candidate for index, candidate in enumerate(candidates) if result.x[index] > 0.5]
    # 排序：正例在前、按分数降序、按录音文件、起始时间、标签排列
    selected.sort(
        key=lambda item: (
            0 if item.target_label == "1" else 1,
            -item.score,
            item.recording_file,
            item.start_seconds,
            item.db_label,
        )
    )
    return selected


def select_samples(audio_index: dict[str, Path] | None = None) -> list[Candidate]:
    """
    主样本选择流程 (第339行以下)：
      1. 从 DB 读取所有候选检测
      2. 过滤掉无对应音频文件的候选（如果提供了 audio_index）
      3. 按来源录音文件分组，每来源取最多 73 个不重叠的高分片段
      4. 对负类候选也要与正例的时间区间不重叠
      5. MILP 优化选出最终平衡集
      6. 验证最终选出的正例、负例计数和每个来源的平衡性
    """
    # 第 1 步：从 DB 读取所有候选
    candidates_by_target: dict[str, list[Candidate]] = {}
    for target in LABEL_TARGETS:
        candidates = read_candidates(
            target["db_label"],
            target["label"],
            target["target_label"],
            target["negative_type"],
        )
        # 可选过滤：只保留有对应音频文件的候选
        if audio_index is not None:
            candidates = [
                candidate
                for candidate in candidates
                if find_audio_file(candidate.recording_file, audio_index) is not None
            ]
        candidates_by_target[target["db_label"]] = candidates

    # 第 2 步：正例候选 -> 按来源分组 -> 去重叠 -> 构成 positive_pool
    positive_target = next(target for target in LABEL_TARGETS if target["target_label"] == "1")
    raw_positive_by_source: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates_by_target[positive_target["db_label"]]:
        raw_positive_by_source[candidate.recording_file].append(candidate)

    positive_pool = {
        source: non_overlapping_candidates(rows, MAX_CLIPS_PER_CLASS_PER_SOURCE)
        for source, rows in raw_positive_by_source.items()
    }
    # 记录正例的时间区间，供负例过滤使用
    positive_intervals = {
        source: [candidate_interval(candidate) for candidate in rows]
        for source, rows in positive_pool.items()
    }

    # 第 3 步：负例候选 -> 按来源分组 -> 过滤掉与正例重叠的 -> 去重叠 -> 构成 negative_pool
    negative_pool: dict[str, list[Candidate]] = defaultdict(list)
    negative_targets = [target for target in LABEL_TARGETS if target["target_label"] == "0"]
    for target in negative_targets:
        by_source: dict[str, list[Candidate]] = defaultdict(list)
        for candidate in candidates_by_target[target["db_label"]]:
            if candidate.recording_file not in positive_pool:
                continue  # 只保留与正例同一来源的负例（保证配对）
            start, end = candidate_interval(candidate)
            if overlaps(start, end, positive_intervals[candidate.recording_file]):
                continue  # 避免与正例的时间区间重叠
            by_source[candidate.recording_file].append(candidate)
        for source, rows in by_source.items():
            negative_pool[source].extend(
                non_overlapping_candidates(rows, MAX_CLIPS_PER_CLASS_PER_SOURCE)
            )

    # 第 4 步：MILP 优化
    selected = optimize_balanced_selection(positive_pool, negative_pool)

    # 第 5 步：验证结果
    class_counts = Counter(candidate.target_label for candidate in selected)
    expected_counts = Counter()
    for target in LABEL_TARGETS:
        expected_counts[target["target_label"]] += target["count"]
    source_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for candidate in selected:
        source_counts[candidate.recording_file][candidate.target_label] += 1
    if class_counts != expected_counts:
        raise AssertionError(f"selection returned unexpected class counts: {dict(class_counts)}")
    if any(counts["1"] != counts["0"] for counts in source_counts.values()):
        raise AssertionError("selection returned a source with unequal positive and negative counts")
    return selected


# ============================================================
# 第 3 阶段：输出文件写入 (第384-477行)
# ============================================================

def write_download_files(selected: list[Candidate]) -> None:
    """
    写出下载清单 CSV：记录每个录音文件中选中了多少片段、包含哪些标签、最高分等。
    按最高分降序排列，方便优先下载高质量录音。
    """
    grouped: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in selected:
        grouped[candidate.recording_file].append(candidate)

    rows = []
    for recording_file, candidates in grouped.items():
        best = max(candidates, key=lambda item: item.score)
        rows.append(
            {
                "recording_file": recording_file,
                "location": best.location,
                "timestamp": best.timestamp,
                "selected_clips": len(candidates),
                "labels": ";".join(sorted({item.db_label for item in candidates})),
                "best_score": best.score,
            }
        )
    rows.sort(key=lambda row: (-row["best_score"], row["recording_file"]))

    DOWNLOAD_FILES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with DOWNLOAD_FILES_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DOWNLOAD_FIELDS)
        writer.writeheader()
        for index, row in enumerate(rows, start=1):
            writer.writerow(
                {
                    "download_order": index,
                    "recording_file": row["recording_file"],
                    "location": row["location"],
                    "timestamp": row["timestamp"],
                    "selected_clips": row["selected_clips"],
                    "labels": row["labels"],
                    "best_score": f"{row['best_score']:.6f}",
                }
            )


def resolve_audio_files(selected: list[Candidate], audio_index: dict[str, Path]) -> dict[str, Path]:
    """
    查找每个选中候选对应的原始音频文件路径。
    如果某个录音文件找不到，直接报错。
    """
    resolved: dict[str, Path] = {}
    for recording_file in sorted({item.recording_file for item in selected}):
        source = find_audio_file(recording_file, audio_index)
        if source is None:
            raise FileNotFoundError(f"missing source recording: {recording_file}")
        resolved[recording_file] = source
    return resolved


def clip_id(candidate: Candidate, rank: int) -> str:
    """
    生成片段 ID：
      正例: storm_petrel_00001_<录音源文件名>_<起始毫秒>
      负例: <negative_type>_00001_<录音源文件名>_<起始毫秒>
    """
    source = Path(candidate.recording_file).stem
    start = int(round(candidate.start_seconds * 1000))
    prefix = "storm_petrel" if candidate.target_label == "1" else candidate.negative_type
    return f"{prefix}_{rank:05d}_{source}_{start:010d}"


def extract_clip(source: Path, destination: Path, start_seconds: float) -> None:
    """
    从原始音频文件切出 3 秒片段并写入 WAV：
      1. librosa.load() 读取单声道、48kHz
      2. offset=start_seconds, duration=3.0
      3. soundfile.write() 写出为 16-bit PCM WAV
    """
    import librosa
    import soundfile as sf

    audio, _ = librosa.load(
        source,
        sr=SAMPLE_RATE,
        mono=True,
        offset=start_seconds,
        duration=CLIP_SECONDS,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    sf.write(destination, audio, SAMPLE_RATE, subtype="PCM_16")


def clean_generated_outputs() -> None:
    """
    清理之前生成的输出目录和文件，防止残留。
    删除:
      - clips/storm_petrel/
      - clips/not_storm_petrel/
      - binary_labels.csv, folds.csv, fold_balance_report.csv
    """
    for folder in (CLIPS_DIR / POSITIVE_FOLDER, CLIPS_DIR / NEGATIVE_FOLDER):
        if folder.exists():
            shutil.rmtree(folder)
    for path in (BINARY_LABELS_PATH, FOLDS_PATH, FOLD_BALANCE_REPORT_PATH):
        if path.exists():
            path.unlink()


def extract_selected_clips(selected: list[Candidate], sources: dict[str, Path]) -> list[dict[str, str]]:
    """
    对每个选中的候选 (第458行以下)：
      1. 清理旧输出
      2. 生成 clip ID
      3. 切取 3 秒 WAV 片段
      4. 构造标签行（待写入 CSV）
    """
    clean_generated_outputs()
    rows: list[dict[str, str]] = []
    for rank, candidate in enumerate(selected, start=1):
        identifier = clip_id(candidate, rank)
        class_folder = POSITIVE_FOLDER if candidate.target_label == "1" else NEGATIVE_FOLDER
        destination = CLIPS_DIR / class_folder / f"{identifier}.wav"
        source = sources[candidate.recording_file]
        extract_clip(source, destination, candidate.start_seconds)
        rows.append(
            {
                "clip_id": identifier,
                "label": candidate.label,
                "target_label": candidate.target_label,
                "negative_type": candidate.negative_type,
                "species": candidate.species,
                "call_type": "",
                "source_file": candidate.recording_file,
                "source_path": str(source),
                "clip_path": str(destination),
                "start_seconds": f"{candidate.start_seconds:.3f}",
                "end_seconds": f"{candidate.start_seconds + CLIP_SECONDS:.3f}",
                "score": f"{candidate.score:.6f}",
                "split": "",
                "outer_fold": "",
                "inner_fold": "",
            }
        )
    return rows


def write_binary_labels(rows: list[dict[str, str]]) -> None:
    """写出二分类标签 CSV 文件（ML 训练用的完整标注表）。"""
    BINARY_LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with BINARY_LABELS_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=BINARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# 第 4 阶段：Grouped Nested Cross-Validation Fold 分配 (第482-667行)
# ============================================================

def planning_rows(selected: list[Candidate]) -> list[dict[str, str]]:
    """从选中候选中提取 fold 规划所需的最小字段集（不涉及文件路径）。"""
    return [
        {
            "source_file": candidate.recording_file,
            "target_label": candidate.target_label,
            "negative_type": candidate.negative_type,
        }
        for candidate in selected
    ]


def group_metrics(rows: list[dict[str, str]]) -> dict[str, Counter[str]]:
    """按来源录音文件统计正负样本数。"""
    metrics: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        values = metrics[row["source_file"]]
        values["total"] += 1
        values["positive"] += row["target_label"] == "1"
    return metrics


def assign_groups_to_folds(rows: list[dict[str, str]], fold_count: int, title: str) -> dict[str, int]:
    """
    用 MILP 将已平衡的来源文件组分配到相同大小的 fold 中 (第496行以下)。
    约束：
      1. 每个来源必须完整分到某个 fold（不能拆分到多个 fold）——这就是 "grouped"
      2. 每个 fold 的大小必须接近 target_size ± 10%
      3. 同时最小化 fold 与 target_size 的偏差之和（L1 距离）
    先决条件：每个来源组的正例数 = 负例数（50:50）
    """
    if not rows:
        return {}

    metrics_by_group = group_metrics(rows)
    groups = sorted(metrics_by_group)
    if len(groups) < fold_count:
        raise FoldAssignmentError(f"{title} needs {fold_count} source groups; found {len(groups)}")
    if any(values["positive"] * 2 != values["total"] for values in metrics_by_group.values()):
        raise FoldAssignmentError(f"{title} requires every source group to be 50:50")

    group_sizes = np.asarray([metrics_by_group[group]["total"] for group in groups], dtype=float)
    group_count = len(groups)
    assignment_variables = group_count * fold_count
    variable_count = assignment_variables + fold_count  # 加上每个 fold 的偏差变量

    # ---- 构建 MILP 约束 ----
    constraint_rows: list[np.ndarray] = []
    lower_bounds: list[float] = []
    upper_bounds: list[float] = []

    def add_constraint(coefficients: np.ndarray, lower: float, upper: float) -> None:
        """向 fold-assignment 约束集合添加一行。"""

        constraint_rows.append(coefficients)
        lower_bounds.append(lower)
        upper_bounds.append(upper)

    def fold_size_coefficients(fold_index: int) -> np.ndarray:
        """返回表示第 fold_index 个 fold 大小的系数向量。"""
        coefficients = np.zeros(variable_count)
        for group_index in range(group_count):
            coefficients[group_index * fold_count + fold_index] = group_sizes[group_index]
        return coefficients

    # 约束 A: 每个来源组只分配到一个 fold（sum of one-hot = 1）
    for group_index in range(group_count):
        coefficients = np.zeros(variable_count)
        start = group_index * fold_count
        coefficients[start : start + fold_count] = 1
        add_constraint(coefficients, 1, 1)

    # 约束 B: 每个 fold 大小必须在 target_size ± 10% 范围内
    target_size = group_sizes.sum() / fold_count
    minimum_size = ceil(target_size * (1 - FOLD_SIZE_TOLERANCE))
    maximum_size = floor(target_size * (1 + FOLD_SIZE_TOLERANCE))
    for fold_index in range(fold_count):
        fold_size = fold_size_coefficients(fold_index)
        add_constraint(fold_size, minimum_size, maximum_size)

        # 偏差变量：fold_size - deviation <= target_size
        #         -fold_size - deviation <= -target_size
        # 等价于 |fold_size - target_size| <= deviation，最小化 sum(deviation)
        deviation_index = assignment_variables + fold_index
        coefficients = fold_size.copy()
        coefficients[deviation_index] = -1
        add_constraint(coefficients, -np.inf, target_size)

        coefficients = -fold_size
        coefficients[deviation_index] = -1
        add_constraint(coefficients, -np.inf, -target_size)

    # ---- 设置变量类型和边界 ----
    variable_lower = np.zeros(variable_count)
    variable_upper = np.full(variable_count, np.inf)
    variable_upper[:assignment_variables] = 1  # 分配变量是 0/1
    integrality = np.zeros(variable_count, dtype=int)
    integrality[:assignment_variables] = 1  # 只有分配变量是整数

    # ---- 优化目标：最小化偏差总和 ----
    objective = np.zeros(variable_count)
    objective[assignment_variables:] = 1
    result = milp(
        c=objective,
        integrality=integrality,
        bounds=Bounds(variable_lower, variable_upper),
        constraints=LinearConstraint(
            np.vstack(constraint_rows),
            np.asarray(lower_bounds),
            np.asarray(upper_bounds),
        ),
        options={"time_limit": 120, "mip_rel_gap": 0.0},
    )
    if result.status != 0 or not result.success or result.x is None:
        raise FoldAssignmentError(
            f"{title} did not reach a proven optimum: status={result.status}, {result.message}"
        )

    # ---- 提取分配结果 ----
    assignments: dict[str, int] = {}
    for group_index, group in enumerate(groups):
        values = result.x[group_index * fold_count : (group_index + 1) * fold_count]
        assignments[group] = int(np.argmax(values)) + 1  # 找到 one-hot 编码中为 1 的位置
    return assignments


def partition_stats(rows: list[dict[str, str]]) -> Counter[str]:
    """计算一个数据分区的统计信息。"""
    stats: Counter[str] = Counter()
    stats["total"] = len(rows)
    stats["positive"] = sum(row["target_label"] == "1" for row in rows)
    stats["negative"] = stats["total"] - stats["positive"]
    stats["environment_noise"] = sum(row.get("negative_type", "") == "environment_noise" for row in rows)
    stats["other_bird"] = sum(row.get("negative_type", "") == "other_bird" for row in rows)
    stats["source_files"] = len({row["source_file"] for row in rows})
    return stats


def validate_partition(rows: list[dict[str, str]], title: str) -> None:
    """验证某个数据分区是否为严格 50:50（不能在训练时引入类别不平衡）。"""
    stats = partition_stats(rows)
    ratio = stats["positive"] / stats["total"] if stats["total"] else 0.0
    if ratio != 0.5:
        raise FoldAssignmentError(f"{title} is not 50:50")
    print(
        f"{title}: positive={stats['positive']}, negative={stats['negative']}, "
        f"total={stats['total']}, source_files={stats['source_files']}, positive_ratio={ratio:.3f}"
    )


def build_fold_plan(rows: list[dict[str, str]]) -> FoldPlan:
    """
    构建 grouped nested CV fold 方案 (第569行以下)：
      1. 将来源组分配到 5 个 outer folds
      2. 对每个 outer fold 的 train 集，进一步分配到 3 个 inner folds
      3. 验证每个 split 都是严格的 50:50
    这是 "Grouped Nested Cross-Validation"——来源组作为一个整体分配到 fold 中，
    确保同一个录音文件的不同片段不会出现在 train 和 test 里。
    """
    outer_assignments = assign_groups_to_folds(rows, OUTER_FOLDS, "outer")
    inner_assignments: dict[int, dict[str, int]] = {}
    for outer_fold in range(1, OUTER_FOLDS + 1):
        outer_test = [row for row in rows if outer_assignments[row["source_file"]] == outer_fold]
        outer_train = [row for row in rows if outer_assignments[row["source_file"]] != outer_fold]
        validate_partition(outer_train, f"outer {outer_fold:02d} train")
        validate_partition(outer_test, f"outer {outer_fold:02d} test")

        assignments = assign_groups_to_folds(outer_train, INNER_FOLDS, f"outer {outer_fold:02d} inner")
        inner_assignments[outer_fold] = assignments
        for inner_fold in range(1, INNER_FOLDS + 1):
            inner_validation = [row for row in outer_train if assignments[row["source_file"]] == inner_fold]
            inner_train = [row for row in outer_train if assignments[row["source_file"]] != inner_fold]
            validate_partition(inner_train, f"outer {outer_fold:02d} inner {inner_fold:02d} train")
            validate_partition(inner_validation, f"outer {outer_fold:02d} inner {inner_fold:02d} validation")
    return FoldPlan(outer_assignments=outer_assignments, inner_assignments=inner_assignments)


def build_fold_rows(rows: list[dict[str, str]], plan: FoldPlan | None = None) -> list[dict[str, str]]:
    """
    根据 fold 方案为每个 outer fold 生成标签行（含 split 和 fold 编号）。
    每个 outer fold 会把所有样本都写一遍，train 行标 inner fold，test 行标 outer_test。
    """
    plan = plan or build_fold_plan(rows)
    output_rows: list[dict[str, str]] = []
    for outer_fold in range(1, OUTER_FOLDS + 1):
        inner_assignments = plan.inner_assignments[outer_fold]
        for row in rows:
            out = {field: row.get(field, "") for field in BINARY_FIELDS}
            out["outer_fold"] = f"{outer_fold:02d}"
            if plan.outer_assignments[row["source_file"]] == outer_fold:
                out["split"] = "outer_test"
                out["inner_fold"] = ""
            else:
                out["split"] = "outer_train"
                out["inner_fold"] = f"{inner_assignments[row['source_file']]:02d}"
            output_rows.append(out)
    return output_rows


def build_balance_report_rows(fold_rows: list[dict[str, str]]) -> list[dict[str, str | int]]:
    """为 fold 方案的每个 split（train/test/validation）生成平衡统计数据。"""
    report_rows: list[dict[str, str | int]] = []

    def append_stats(outer_fold: str, inner_fold: str, split: str, rows: list[dict[str, str]]) -> None:
        """计算并追加一个 nested-CV 分区的平衡统计。"""

        stats = partition_stats(rows)
        report_rows.append(
            {
                "outer_fold": outer_fold,
                "inner_fold": inner_fold,
                "split": split,
                "total": stats["total"],
                "positive": stats["positive"],
                "negative": stats["negative"],
                "positive_ratio": f"{stats['positive'] / stats['total']:.6f}",
                "environment_noise": stats["environment_noise"],
                "other_bird": stats["other_bird"],
                "source_files": stats["source_files"],
            }
        )

    for outer_fold in sorted({row["outer_fold"] for row in fold_rows}):
        outer_rows = [row for row in fold_rows if row["outer_fold"] == outer_fold]
        outer_train = [row for row in outer_rows if row["split"] == "outer_train"]
        outer_test = [row for row in outer_rows if row["split"] == "outer_test"]
        append_stats(outer_fold, "", "train", outer_train)
        append_stats(outer_fold, "", "test", outer_test)
        for inner_fold in sorted({row["inner_fold"] for row in outer_train}):
            inner_train = [row for row in outer_train if row["inner_fold"] != inner_fold]
            inner_validation = [row for row in outer_train if row["inner_fold"] == inner_fold]
            append_stats(outer_fold, inner_fold, "train", inner_train)
            append_stats(outer_fold, inner_fold, "validation", inner_validation)
    return report_rows


def write_folds(rows: list[dict[str, str]], plan: FoldPlan | None = None) -> None:
    """写出 fold 分配 CSV 和平衡报告 CSV。"""
    fold_rows = build_fold_rows(rows, plan)
    FOLDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with FOLDS_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=BINARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(fold_rows)

    with FOLD_BALANCE_REPORT_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=BALANCE_REPORT_FIELDS)
        writer.writeheader()
        writer.writerows(build_balance_report_rows(fold_rows))
    print(f"已写出 fold 平衡报告 {FOLD_BALANCE_REPORT_PATH}")


def print_summary(rows: list[dict[str, str]]) -> None:
    """打印最终的摘要统计。"""
    target_counts = Counter(row["target_label"] for row in rows)
    negative_counts = Counter(row["negative_type"] for row in rows if row["target_label"] == "0")
    print(f"已写出二分类标签表 {BINARY_LABELS_PATH}")
    print(f"正例: {target_counts['1']}; 负例: {target_counts['0']}")
    print(f"负例类型: {dict(sorted(negative_counts.items()))}")
    print(f"已写出 grouped nested CV fold 表 {FOLDS_PATH}")


# ============================================================
# 主流程
# ============================================================
def main() -> None:
    """
    完整 pipeline (第676-695行)：
      1. 打印项目配置
      2. 索引音频文件
      3. 选择平衡样本（MILP）
      4. 预跑 fold 分配验证可行性
      5. 写出下载清单
      6. 解析音频文件路径
      7. 切出 3 秒 WAV 片段
      8. 写出二分类标签 CSV
      9. 写出 fold 分配 CSV 和平衡报告
      10. 打印摘要
    """
    print_config()
    audio_index = index_existing_audio_files()
    selected = select_samples(audio_index)
    # preflight: 在切出实际文件前先做 fold 分配，确认可行性
    print("\nPreflight grouped fold optimization (no generated output has been removed yet).")
    fold_plan = build_fold_plan(planning_rows(selected))
    write_download_files(selected)
    print(f"已写出所需录音清单 {DOWNLOAD_FILES_PATH}")

    sources = resolve_audio_files(selected, audio_index)
    rows = extract_selected_clips(selected, sources)
    write_binary_labels(rows)
    write_folds(rows, fold_plan)
    print_summary(rows)


if __name__ == "__main__":
    main()
