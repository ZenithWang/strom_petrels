from __future__ import annotations

import importlib
import importlib.metadata
import os
import platform
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_DIR = Path(os.environ.get("STORM_PETREL_WORKSPACE_DIR", PROJECT_ROOT / "workspace"))
CACHE_DIR = WORKSPACE_DIR / "environment_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


REQUIRED_PACKAGES = {
    "birdnet-analyzer": "birdnet_analyzer",
    "librosa": "librosa",
    "matplotlib": "matplotlib",
    "numpy": "numpy",
    "pandas": "pandas",
    "scikit-learn": "sklearn",
    "scipy": "scipy",
    "soundfile": "soundfile",
    "tensorflow": "tensorflow",
}


def main() -> int:
    print(f"Python: {sys.version.split()[0]}")
    print(f"Interpreter: {sys.executable}")
    print(f"Platform: {platform.platform()}")

    failed = False
    for distribution, module in REQUIRED_PACKAGES.items():
        try:
            version = importlib.metadata.version(distribution)
            importlib.import_module(module)
        except Exception as exc:  # Import errors may include missing native libraries.
            failed = True
            print(f"[FAIL] {distribution}: {type(exc).__name__}: {exc}")
        else:
            print(f"[ OK ] {distribution} {version}")

    if sys.version_info[:2] != (3, 11):
        failed = True
        print("[FAIL] This project environment must use Python 3.11.")

    if failed:
        print("Environment verification failed.")
        return 1

    print("Environment verification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
