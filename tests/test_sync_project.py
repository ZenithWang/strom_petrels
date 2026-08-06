from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sync_project import SYNC_DIRECTORIES, SYNC_FILES, build_commands, destination


def test_paths_are_identical_on_both_sides() -> None:
    target = "alyzw89@strange:~/working/"
    assert destination(target, "project_config.py") == "alyzw89@strange:~/working/project_config.py"
    assert destination(target, "comp_chick_adult", directory=True) == (
        "alyzw89@strange:~/working/comp_chick_adult/"
    )


def test_default_is_safe_dry_run_with_runtime_data_excluded() -> None:
    commands = build_commands("alyzw89@strange:~/working", apply=False, delete=False)
    assert len(commands) == len(SYNC_FILES) + len(SYNC_DIRECTORIES)
    assert all("--dry-run" in command for command in commands)
    assert all("--delete" not in command for command in commands)
    joined = "\n".join(" ".join(command) for command in commands)
    for forbidden in ("*.db", "*.wav", "*.tflite", "__pycache__/"):
        assert forbidden in joined
    assert "workspace" not in {Path(command[-1]).name for command in commands}


def test_delete_is_limited_to_functional_directories() -> None:
    commands = build_commands("server:/working", apply=True, delete=True)
    file_commands = commands[: len(SYNC_FILES)]
    directory_commands = commands[len(SYNC_FILES) :]
    assert all("--delete" not in command for command in file_commands)
    assert all("--delete" in command for command in directory_commands)
