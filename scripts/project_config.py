from __future__ import annotations

import os
from pathlib import Path


def _path_from_env(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, default)).expanduser().resolve()


PROJECT_ROOT = _path_from_env("STORM_PETREL_PROJECT_ROOT", Path(__file__).resolve().parents[1])
DB_PATH = _path_from_env(
    "STORM_PETREL_DB_PATH",
    PROJECT_ROOT / "scripts" / "data_processing" / "storm_petrel_2021_full.db",
)
AUDIO_ROOT = _path_from_env("STORM_PETREL_AUDIO_ROOT", PROJECT_ROOT / "download" / "data")
WORKSPACE_DIR = _path_from_env("STORM_PETREL_WORKSPACE_DIR", PROJECT_ROOT / "workspace")
CLIPS_DIR = WORKSPACE_DIR / "clips"
DOWNLOAD_FILES_PATH = WORKSPACE_DIR / "download_files.csv"
BINARY_LABELS_PATH = WORKSPACE_DIR / "binary_labels.csv"
FOLDS_PATH = WORKSPACE_DIR / "nested_cv_folds.csv"
DATASET_ROOT = WORKSPACE_DIR / "nested_cv_datasets"
MODEL_ROOT = _path_from_env("STORM_PETREL_MODEL_ROOT", WORKSPACE_DIR / "models" / "nested_cv")
NUMBA_CACHE_DIR = WORKSPACE_DIR / "numba_cache"
TMP_DIR = WORKSPACE_DIR / "tmp"


def print_config() -> None:
    rows = [
        ("PROJECT_ROOT", PROJECT_ROOT),
        ("DB_PATH", DB_PATH),
        ("AUDIO_ROOT", AUDIO_ROOT),
        ("WORKSPACE_DIR", WORKSPACE_DIR),
        ("CLIPS_DIR", CLIPS_DIR),
        ("FOLDS_PATH", FOLDS_PATH),
        ("DATASET_ROOT", DATASET_ROOT),
        ("MODEL_ROOT", MODEL_ROOT),
        ("NUMBA_CACHE_DIR", NUMBA_CACHE_DIR),
        ("TMP_DIR", TMP_DIR),
    ]
    print("Resolved project paths:")
    for name, path in rows:
        print(f"  {name}: {path}")
