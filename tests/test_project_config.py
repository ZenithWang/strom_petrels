from __future__ import annotations

import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_config import _default_project_root, _load_env_file


def test_default_project_root_for_shared_flat_layout() -> None:
    config = Path("/srv/storm_petrels/project_config.py")
    assert _default_project_root(config) == Path("/srv/storm_petrels")


def test_env_file_expands_values_and_preserves_shell_precedence(tmp_path: Path) -> None:
    names = [
        "STORM_TEST_ROOT",
        "STORM_TEST_WORKSPACE",
        "STORM_TEST_EXISTING",
    ]
    original = {name: os.environ.get(name) for name in names}
    try:
        os.environ.pop("STORM_TEST_ROOT", None)
        os.environ.pop("STORM_TEST_WORKSPACE", None)
        os.environ["STORM_TEST_EXISTING"] = "from-shell"
        env_file = tmp_path / ".env"
        env_file.write_text(
            "\n".join(
                [
                    "# test values",
                    f"STORM_TEST_ROOT={tmp_path}",
                    "STORM_TEST_WORKSPACE=${STORM_TEST_ROOT}/workspace",
                    "export STORM_TEST_EXISTING=from-file",
                ]
            ),
            encoding="utf-8",
        )
        _load_env_file(env_file)
        assert os.environ["STORM_TEST_ROOT"] == str(tmp_path)
        assert os.environ["STORM_TEST_WORKSPACE"] == f"{tmp_path}/workspace"
        assert os.environ["STORM_TEST_EXISTING"] == "from-shell"
    finally:
        for name, value in original.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
