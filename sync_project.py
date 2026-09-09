"""以相同目录结构把轻量项目源码安全同步到服务器。"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parent
SYNC_DIRECTORIES = (
    "data_processing",
    "model_training",
    "long_recording_analysis",
    "comp_chick_adult",
    "results_analysis",
    "chick_screening",
)
SYNC_FILES = (
    "project_config.py",
    "verify_environment.py",
    "environment.yml",
    ".env.example",
    "README.md",
    "README_zh.md",
    "sync_project.py",
)
EXCLUDES = (
    "__pycache__/",
    "*.pyc",
    "*.pyo",
    ".ipynb_checkpoints/",
    "*.db",
    "*.sqlite",
    "*.sqlite3",
    "*.bak",
    "*.wav",
    "*.mp3",
    "*.flac",
    "*.ogg",
    "*.m4a",
    "*.tflite",
    "*.h5",
    "*.keras",
    "*.pkl",
    "*.joblib",
)


def destination(target_root: str, name: str, directory: bool = False) -> str:
    """构造远端文件或目录目标路径。"""

    return f"{target_root.rstrip('/')}/{name}{'/' if directory else ''}"


def rsync_base(apply: bool) -> list[str]:
    """构建包含安全排除规则的 rsync 基础命令。"""

    command = ["rsync", "-av", "--itemize-changes"]
    if not apply:
        command.append("--dry-run")
    for pattern in EXCLUDES:
        command.extend(["--exclude", pattern])
    return command


def build_commands(target_root: str, apply: bool, delete: bool) -> list[list[str]]:
    """为根文件和功能目录生成独立的 rsync 命令。"""

    if not target_root.strip():
        raise ValueError("target_root must not be empty")
    base = rsync_base(apply)
    commands: list[list[str]] = []
    for name in SYNC_FILES:
        source = PROJECT_ROOT / name
        if not source.is_file():
            raise FileNotFoundError(f"Required synchronized file not found: {source}")
        commands.append([*base, str(source), destination(target_root, name)])
    for name in SYNC_DIRECTORIES:
        source = PROJECT_ROOT / name
        if not source.is_dir():
            raise FileNotFoundError(f"Required synchronized directory not found: {source}")
        command = [*base]
        if delete:
            command.append("--delete")
        command.extend([f"{source}/", destination(target_root, name, directory=True)])
        commands.append(command)
    return commands


def run_sync(target_root: str, apply: bool = False, delete: bool = False) -> None:
    """预览或执行源码同步，不传输数据和模型。"""

    if shutil.which("rsync") is None:
        raise FileNotFoundError("rsync is required but was not found on PATH")
    mode = "APPLY" if apply else "DRY RUN"
    print(f"Same-layout synchronization mode: {mode}", flush=True)
    print(f"Target: {target_root}", flush=True)
    print(
        "workspace, .env, databases, audio, models, caches, tests, and local tooling are not synchronized.",
        flush=True,
    )
    for command in build_commands(target_root, apply, delete):
        subprocess.run(command, check=True)
    if apply:
        print("Project source synchronization complete.")
    else:
        print("Dry run complete. Re-run with --apply to synchronize.")


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    """解析远端目标、实际执行和清理陈旧源码选项。"""

    parser = argparse.ArgumentParser(
        description="Synchronize the workflow while preserving its repository-relative layout."
    )
    parser.add_argument("target", help="For example: username@server:/path/to/strom_petrels")
    parser.add_argument("--apply", action="store_true", help="Perform synchronization; default is dry-run.")
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete stale files inside the five synchronized code directories; requires --apply.",
    )
    parsed = parser.parse_args(args)
    if parsed.delete and not parsed.apply:
        parser.error("--delete requires --apply")
    return parsed


def main() -> None:
    """运行源码同步命令行入口。"""

    args = parse_args()
    run_sync(args.target, args.apply, args.delete)


if __name__ == "__main__":
    main()
