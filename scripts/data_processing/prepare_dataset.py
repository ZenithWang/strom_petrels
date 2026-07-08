from __future__ import annotations

import csv
import shutil
import sqlite3
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from sklearn.model_selection import StratifiedGroupKFold

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from project_config import (
    AUDIO_ROOT,
    BINARY_LABELS_PATH,
    CLIPS_DIR,
    DB_PATH,
    DOWNLOAD_FILES_PATH,
    FOLDS_PATH,
    print_config,
)
SAMPLE_RATE = 48000
CLIP_SECONDS = 3.0
AUDIO_EXTENSIONS = (".flac", ".webm", ".wav", ".mp3", ".ogg", ".m4a")

POSITIVE_FOLDER = "storm_petrel"
NEGATIVE_FOLDER = "not_storm_petrel"
OUTER_FOLDS = 5
INNER_FOLDS = 3

LABEL_TARGETS = (
    {
        "db_label": "Hydrobates pelagicus_European Storm-Petrel",
        "count": 500,
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

BINARY_FIELDS = [
    "clip_id",
    "label",
    "target_label",
    "negative_type",
    "species",
    "call_type",
    "source_file",
    "source_path",
    "clip_path",
    "start_seconds",
    "end_seconds",
    "score",
    "split",
    "outer_fold",
    "inner_fold",
]

DOWNLOAD_FIELDS = [
    "download_order",
    "recording_file",
    "location",
    "timestamp",
    "selected_clips",
    "labels",
    "best_score",
]


@dataclass(frozen=True)
class Candidate:
    db_label: str
    species: str
    score: float
    location: int
    timestamp: str
    recording_file: str
    duration_seconds: float
    start_seconds: float
    end_seconds: float
    label: str
    target_label: str
    negative_type: str


def parse_detection_desc(desc: str) -> tuple[str, float]:
    species, raw_score = desc.rsplit(" ", 1)
    return species.strip(), float(raw_score)


def read_candidates(db_label: str, label: str, target_label: str, negative_type: str) -> list[Candidate]:
    if not DB_PATH.is_file():
        raise FileNotFoundError(f"database not found: {DB_PATH}")

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
    rows: list[Candidate] = []
    with sqlite3.connect(DB_PATH) as connection:
        for row in connection.execute(query, (f"{db_label} %",)):
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
    rows.sort(
        key=lambda item: (
            -item.score,
            item.location,
            item.timestamp,
            item.start_seconds,
            item.recording_file,
        )
    )
    ##先按 score 排序，score 高的在前面；再按 location、timestamp、start_seconds、recording_file 排序，保证同一录音文件的片段按时间顺序排列
    return rows


def overlaps(start: float, end: float, intervals: list[tuple[float, float]]) -> bool:
    return any(start < interval_end and end > interval_start for interval_start, interval_end in intervals)


def index_existing_audio_files() -> dict[str, Path]:
    if not AUDIO_ROOT.is_dir():
        raise FileNotFoundError(f"audio root not found: {AUDIO_ROOT}")

    indexed: dict[str, Path] = {}
    for path in sorted(item for item in AUDIO_ROOT.rglob("*") if item.is_file()):
        indexed.setdefault(path.name, path)
        indexed.setdefault(path.stem, path)
    return indexed


def find_audio_file(recording_file: str, audio_index: dict[str, Path] | None = None) -> Path | None:
    if audio_index is not None:
        exact = audio_index.get(Path(recording_file).name)
        if exact is not None:
            return exact
        stem_match = audio_index.get(Path(recording_file).stem)
        if stem_match is not None:
            return stem_match

    direct = AUDIO_ROOT / recording_file
    if direct.is_file():
        return direct
    matches = sorted(AUDIO_ROOT.rglob(recording_file)) if AUDIO_ROOT.is_dir() else []
    if matches:
        return matches[0]

    stem = Path(recording_file).stem
    for extension in AUDIO_EXTENSIONS:
        candidate = f"{stem}{extension}"
        direct = AUDIO_ROOT / candidate
        if direct.is_file():
            return direct
        matches = sorted(AUDIO_ROOT.rglob(candidate)) if AUDIO_ROOT.is_dir() else []
        if matches:
            return matches[0]
    return None


def select_samples(audio_index: dict[str, Path] | None = None) -> list[Candidate]:
    selected: list[Candidate] = []
    positive_windows: dict[str, list[tuple[float, float]]] = defaultdict(list)
    used_windows: set[tuple[str, str, str]] = set()

    for target in LABEL_TARGETS:
        candidates = read_candidates(
            target["db_label"],
            target["label"],
            target["target_label"],
            target["negative_type"],
        )
        if audio_index is not None:
            candidates = [
                candidate
                for candidate in candidates
                if find_audio_file(candidate.recording_file, audio_index) is not None
            ]
        target_rows: list[Candidate] = []
        for candidate in candidates:
            start_text = f"{candidate.start_seconds:.3f}"
            end_text = f"{candidate.end_seconds:.3f}"
            window_key = (candidate.recording_file, start_text, end_text)
            if window_key in used_windows:
                continue
            if candidate.target_label == "0" and overlaps(
                candidate.start_seconds,
                candidate.end_seconds,
                positive_windows[candidate.recording_file],
            ):
                continue

            target_rows.append(candidate)
            selected.append(candidate)
            used_windows.add(window_key)
            if candidate.target_label == "1":
                positive_windows[candidate.recording_file].append(
                    (candidate.start_seconds, candidate.end_seconds)
                )
            if len(target_rows) == target["count"]:
                break

        if len(target_rows) < target["count"]:
            raise RuntimeError(
                f"only selected {len(target_rows)} clips for {target['db_label']} "
                f"from recordings that exist under {AUDIO_ROOT}; needed {target['count']}"
            )

    return selected


def write_download_files(selected: list[Candidate]) -> None:
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


def resolve_audio_files(selected: list[Candidate], audio_index: dict[str, Path] | None = None) -> dict[str, Path]:
    resolved: dict[str, Path] = {}
    missing: list[str] = []
    for recording_file in sorted({item.recording_file for item in selected}):
        source = find_audio_file(recording_file, audio_index)
        if source is None:
            missing.append(recording_file)
        else:
            resolved[recording_file] = source

    if missing:
        preview = "\n".join(f"  {name}" for name in missing[:20])
        raise FileNotFoundError(
            f"missing {len(missing)} source recordings under {AUDIO_ROOT}. "
            f"Wrote required file list to {DOWNLOAD_FILES_PATH}.\n{preview}"
        )
    return resolved


def clip_id(candidate: Candidate, rank: int) -> str:
    source = Path(candidate.recording_file).stem
    start = int(round(candidate.start_seconds * 1000))
    prefix = "storm_petrel" if candidate.target_label == "1" else candidate.negative_type
    return f"{prefix}_{rank:05d}_{source}_{start:010d}"


def extract_clip(source: Path, destination: Path, start_seconds: float) -> None:
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
    for folder in (CLIPS_DIR / POSITIVE_FOLDER, CLIPS_DIR / NEGATIVE_FOLDER):
        if folder.exists():
            shutil.rmtree(folder)
    for path in (BINARY_LABELS_PATH, FOLDS_PATH):
        if path.exists():
            path.unlink()


def extract_selected_clips(selected: list[Candidate], sources: dict[str, Path]) -> list[dict[str, str]]:
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
    BINARY_LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with BINARY_LABELS_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=BINARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def group_counts(rows: list[dict[str, str]]) -> dict[str, Counter[str]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        counts[row["source_file"]][row["target_label"]] += 1
    return counts


def assign_groups_to_folds(rows: list[dict[str, str]], fold_count: int, title: str) -> dict[str, int]:
    if not rows:
        return {}

    groups = sorted({row["source_file"] for row in rows})
    actual_folds = min(fold_count, len(groups))
    if actual_folds < fold_count:
        print(f"source_file 只有 {len(groups)} 个，{title} 实际使用 {actual_folds} 折。")
    if actual_folds < 2:
        raise ValueError(f"{title} 至少需要 2 个 source_file 才能分组交叉验证。")

    y = [int(row["target_label"]) for row in rows]
    source_groups = [row["source_file"] for row in rows]
    splitter = StratifiedGroupKFold(n_splits=actual_folds, shuffle=True, random_state=42)
    assignments: dict[str, int] = {}
    for fold_index, (_, test_indices) in enumerate(splitter.split(rows, y, source_groups), start=1):
        for row_index in test_indices:
            assignments[rows[row_index]["source_file"]] = fold_index

    missing = set(groups) - set(assignments)
    if missing:
        raise RuntimeError(f"{title} fold 分配遗漏了 {len(missing)} 个 source_file。")
    return assignments


def print_fold_balance(rows: list[dict[str, str]], assignments: dict[str, int], title: str) -> None:
    stats_by_fold: dict[int, Counter[str]] = defaultdict(Counter)
    sources_by_fold: dict[int, set[str]] = defaultdict(set)
    for row in rows:
        fold = assignments.get(row["source_file"])
        if fold is not None:
            stats_by_fold[fold][row["target_label"]] += 1
            sources_by_fold[fold].add(row["source_file"])

    total_positive = sum(1 for row in rows if row["target_label"] == "1")
    total_count = len(rows)
    target_ratio = total_positive / total_count if total_count else 0.0
    for fold in sorted(stats_by_fold):
        stats = stats_by_fold[fold]
        positive = stats["1"]
        negative = stats["0"]
        total = positive + negative
        ratio = positive / total if total else 0.0
        if positive == 0 or negative == 0:
            raise ValueError(f"{title} {fold:02d} 缺少正类或负类: positive={positive}, negative={negative}")
        marker = ""
        if abs(ratio - target_ratio) > 0.20:
            marker = "  注意：受 source_file 不可拆分限制，类别比例偏离整体比例较大"
        print(
            f"{title} {fold:02d}: positive={positive}, negative={negative}, "
            f"total={total}, source_files={len(sources_by_fold[fold])}, positive_ratio={ratio:.3f}{marker}"
        )


def build_fold_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    outer_assignments = assign_groups_to_folds(rows, OUTER_FOLDS, "outer")
    print_fold_balance(rows, outer_assignments, "outer")
    outer_fold_numbers = sorted(set(outer_assignments.values()))

    output_rows: list[dict[str, str]] = []
    for outer_fold in outer_fold_numbers:
        train_rows = [row for row in rows if outer_assignments.get(row["source_file"]) != outer_fold]
        inner_assignments = assign_groups_to_folds(train_rows, INNER_FOLDS, f"outer {outer_fold:02d} inner")
        print_fold_balance(train_rows, inner_assignments, f"outer {outer_fold:02d} inner")

        for row in rows:
            out = {field: row.get(field, "") for field in BINARY_FIELDS}
            out["outer_fold"] = f"{outer_fold:02d}"
            if outer_assignments.get(row["source_file"]) == outer_fold:
                out["split"] = "outer_test"
                out["inner_fold"] = ""
            else:
                out["split"] = "outer_train"
                out["inner_fold"] = f"{inner_assignments.get(row['source_file'], 1):02d}"
            output_rows.append(out)
    return output_rows


def write_folds(rows: list[dict[str, str]]) -> None:
    fold_rows = build_fold_rows(rows)
    FOLDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with FOLDS_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=BINARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(fold_rows)


def print_summary(rows: list[dict[str, str]]) -> None:
    target_counts = Counter(row["target_label"] for row in rows)
    negative_counts = Counter(row["negative_type"] for row in rows if row["target_label"] == "0")
    print(f"已写出二分类标签表 {BINARY_LABELS_PATH}")
    print(f"正例: {target_counts['1']}; 负例: {target_counts['0']}")
    print(f"负例类型: {dict(sorted(negative_counts.items()))}")
    print(f"已写出 grouped nested CV fold 表 {FOLDS_PATH}")


def main() -> None:
    print_config()
    audio_index = index_existing_audio_files()
    selected = select_samples(audio_index)
    write_download_files(selected)
    print(f"已写出所需录音清单 {DOWNLOAD_FILES_PATH}")

    sources = resolve_audio_files(selected, audio_index)
    rows = extract_selected_clips(selected, sources)
    write_binary_labels(rows)
    write_folds(rows)
    print_summary(rows)


if __name__ == "__main__":
    main()

