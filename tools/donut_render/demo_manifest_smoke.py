from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    sample_script = repo_root / "samples" / "DonutRenderPyDemo" / "donut_render_demo_v0_1.py"

    parser = argparse.ArgumentParser(description="Run the DonutRenderPy demo and validate its output manifest.")
    parser.add_argument("--module-dir", type=Path, default=repo_root / "bin" / "windows-x64")
    parser.add_argument("--runtime-dir", type=Path, default=repo_root)
    parser.add_argument("--output-dir", type=Path, default=repo_root / ".temp" / "demo_manifest_smoke")
    parser.add_argument("--frames", type=int, default=2)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=96)
    args = parser.parse_args()

    command = [
        sys.executable,
        str(sample_script),
        "--module-dir",
        str(args.module_dir),
        "--runtime-dir",
        str(args.runtime_dir),
        "--output-dir",
        str(args.output_dir),
        "--frames",
        str(max(1, int(args.frames))),
        "--width",
        str(max(1, int(args.width))),
        "--height",
        str(max(1, int(args.height))),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    manifest_path = Path(result.stdout.strip().splitlines()[-1])

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_frames = max(1, int(args.frames))
    if manifest.get("frame_count") != expected_frames:
        raise AssertionError(f"Expected frame_count={expected_frames}, got {manifest.get('frame_count')}.")

    frames = manifest.get("frames", [])
    if len(frames) != expected_frames:
        raise AssertionError(f"Expected {expected_frames} frame entries, got {len(frames)}.")

    missing = [entry["path"] for entry in frames if not Path(entry["path"]).is_file()]
    if missing:
        raise AssertionError(f"Missing demo output files: {missing}")

    print(f"demo_manifest_smoke ok: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
