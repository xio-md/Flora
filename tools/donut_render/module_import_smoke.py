from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path


def _default_module_dir(repo_root: Path) -> Path:
    platform_dir = "windows-x64" if os.name == "nt" else "linux-x64"
    return repo_root / "bin" / platform_dir


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser(description="Import the native Donut renderer module from a build output directory.")
    parser.add_argument("--module-dir", type=Path, default=_default_module_dir(repo_root))
    args = parser.parse_args()

    sys.path.insert(0, str(args.module_dir))

    module = None
    for name in ("DonutRenderPyNative", "RtxRenderPy"):
        try:
            module = importlib.import_module(name)
            print(name)
            break
        except ImportError:
            continue

    if module is None:
        raise SystemExit("Failed to import DonutRenderPyNative or RtxRenderPy from the module directory.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
