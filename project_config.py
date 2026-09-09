"""集中解析本地与服务器共用的项目路径配置。"""

from __future__ import annotations

import os
import re
from pathlib import Path


ENV_NAME_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _load_env_file(path: Path) -> None:
    """Load simple KEY=VALUE entries without overriding the process environment."""
    if not path.is_file():
        return
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            raise ValueError(f"Invalid environment entry at {path}:{line_number}")
        name, value = line.split("=", 1)
        name = name.strip()
        value = value.strip()
        if not ENV_NAME_PATTERN.fullmatch(name):
            raise ValueError(f"Invalid environment variable name at {path}:{line_number}: {name!r}")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(name, os.path.expandvars(value))


def _path_from_env(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, default)).expanduser().resolve()


def _default_project_root(config_path: Path) -> Path:
    """Resolve the project root from the shared root-level config file."""
    return config_path.resolve().parent


CONFIG_PATH = Path(__file__).resolve()
DEFAULT_PROJECT_ROOT = _default_project_root(CONFIG_PATH)
ENV_FILE = Path(
    os.environ.get("STORM_PETREL_ENV_FILE", DEFAULT_PROJECT_ROOT / ".env")
).expanduser().resolve()
_load_env_file(ENV_FILE)
PROJECT_ROOT = _path_from_env("STORM_PETREL_PROJECT_ROOT", _default_project_root(CONFIG_PATH))
CODE_ROOT = PROJECT_ROOT
DB_PATH = _path_from_env(
    "STORM_PETREL_DB_PATH",
    CODE_ROOT / "data_processing" / "storm_petrel_2021_full.db",
)
AUDIO_ROOT = _path_from_env("STORM_PETREL_AUDIO_ROOT", PROJECT_ROOT / "download" / "data")
WORKSPACE_DIR = _path_from_env("STORM_PETREL_WORKSPACE_DIR", PROJECT_ROOT / "workspace")
CLIPS_DIR = WORKSPACE_DIR / "clips"
DOWNLOAD_FILES_PATH = WORKSPACE_DIR / "download_files.csv"
BINARY_LABELS_PATH = WORKSPACE_DIR / "binary_labels.csv"
FOLDS_PATH = WORKSPACE_DIR / "nested_cv_folds.csv"
FOLD_BALANCE_REPORT_PATH = WORKSPACE_DIR / "fold_balance_report.csv"
DATASET_ROOT = WORKSPACE_DIR / "nested_cv_datasets"
MODEL_ROOT = _path_from_env("STORM_PETREL_MODEL_ROOT", WORKSPACE_DIR / "models" / "nested_cv")
DEPLOYMENT_MODEL_DIR = _path_from_env(
    "STORM_PETREL_DEPLOYMENT_MODEL_DIR",
    WORKSPACE_DIR / "models" / "deployment",
)
LONG_RECORDING_ANALYSIS_DIR = _path_from_env(
    "STORM_PETREL_LONG_RECORDING_ANALYSIS_DIR",
    WORKSPACE_DIR / "long_recording_analysis",
)
COMP_CHICK_ADULT_DIR = _path_from_env(
    "STORM_PETREL_COMP_CHICK_ADULT_DIR",
    WORKSPACE_DIR / "comp_chick_adult",
)
CHICK_SCREENING_DIR = _path_from_env(
    "STORM_PETREL_CHICK_SCREENING_DIR",
    WORKSPACE_DIR / "chick_screening",
)
NUMBA_CACHE_DIR = WORKSPACE_DIR / "numba_cache"
TMP_DIR = WORKSPACE_DIR / "tmp"


def print_config() -> None:
    """打印当前进程实际解析出的全部关键路径。"""

    rows = [
        ("PROJECT_ROOT", PROJECT_ROOT),
        ("DB_PATH", DB_PATH),
        ("AUDIO_ROOT", AUDIO_ROOT),
        ("WORKSPACE_DIR", WORKSPACE_DIR),
        ("CLIPS_DIR", CLIPS_DIR),
        ("FOLDS_PATH", FOLDS_PATH),
        ("FOLD_BALANCE_REPORT_PATH", FOLD_BALANCE_REPORT_PATH),
        ("DATASET_ROOT", DATASET_ROOT),
        ("MODEL_ROOT", MODEL_ROOT),
        ("DEPLOYMENT_MODEL_DIR", DEPLOYMENT_MODEL_DIR),
        ("LONG_RECORDING_ANALYSIS_DIR", LONG_RECORDING_ANALYSIS_DIR),
        ("COMP_CHICK_ADULT_DIR", COMP_CHICK_ADULT_DIR),
        ("CHICK_SCREENING_DIR", CHICK_SCREENING_DIR),
        ("NUMBA_CACHE_DIR", NUMBA_CACHE_DIR),
        ("TMP_DIR", TMP_DIR),
    ]
    print("Resolved project paths:")
    for name, path in rows:
        print(f"  {name}: {path}")
