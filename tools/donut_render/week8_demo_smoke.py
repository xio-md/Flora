from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    sample_script = repo_root / "samples" / "DonutRenderPyDemo" / "donut_render_demo_v0_5.py"

    parser = argparse.ArgumentParser(description="Run the Week 8 DonutRenderPy demo v0.5 and validate incremental frame metadata.")
    parser.add_argument("--module-dir", type=Path, default=repo_root / "bin" / "windows-x64")
    parser.add_argument("--runtime-dir", type=Path, default=repo_root)
    parser.add_argument("--output-dir", type=Path, default=repo_root / ".temp" / "week8_demo_smoke")
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--height", type=int, default=120)
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
    if manifest.get("demo") != "DonutRenderPy Demo v0.5":
        raise AssertionError(f"Unexpected demo label: {manifest.get('demo')}.")
    if manifest.get("frame_count") != expected_frames:
        raise AssertionError(f"Expected frame_count={expected_frames}, got {manifest.get('frame_count')}.")

    frames = manifest.get("frames", [])
    if len(frames) != expected_frames:
        raise AssertionError(f"Expected {expected_frames} frame entries, got {len(frames)}.")

    missing = [entry["path"] for entry in frames if not Path(entry["path"]).is_file()]
    if missing:
        raise AssertionError(f"Missing demo output files: {missing}")

    if not frames or frames[0].get("report_mode") != "full_rebuild":
        raise AssertionError("The first demo frame should bootstrap through full_rebuild.")

    incremental_modes = [entry.get("report_mode") for entry in frames[1:] if str(entry.get("report_mode", "")).startswith("incremental")]
    if len(incremental_modes) != max(0, expected_frames - 1):
        raise AssertionError("Every post-bootstrap frame should stay on an incremental update path.")

    required_ops = {"apply_environment", "apply_camera", "apply_rigid_transforms", "advance_scene"}
    for entry in frames[1:]:
        operations = set(entry.get("operations", []))
        missing_ops = sorted(required_ops.difference(operations))
        if missing_ops:
            raise AssertionError(f"Frame {entry.get('frame_index')} missing operations {missing_ops}.")

    if int(manifest.get("incremental_frame_count", 0)) != max(0, expected_frames - 1):
        raise AssertionError(
            f"Expected incremental_frame_count={max(0, expected_frames - 1)}, got {manifest.get('incremental_frame_count')}."
        )

    print(f"week8_demo_smoke ok: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
