from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from python_demo_common import default_output_dir, frame_output_path, write_rgba_bytes_ppm


def _import_native_renderer(module_dir: Path):
    sys.path.insert(0, str(module_dir))
    try:
        import DonutRenderPyNative as rr
        return rr
    except ImportError:
        import RtxRenderPy as rr
        return rr


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser(description="Render one offscreen PBR frame with the native Donut renderer module.")
    parser.add_argument("--module-dir", type=Path, default=repo_root / "bin" / "windows-x64")
    parser.add_argument("--scene", type=Path,
                        default=repo_root / "external" / "donut" / "thirdparty" / "cgltf" / "fuzz" / "data" / "Box.glb")
    parser.add_argument("--output", type=Path, default=None, help="Legacy direct output path. Overrides --output-dir/output-stem.")
    parser.add_argument("--output-dir", type=Path, default=default_output_dir(repo_root, "headless_pbr_py"))
    parser.add_argument("--output-stem", type=str, default="headless_pbr_frame")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    args = parser.parse_args()

    rr = _import_native_renderer(args.module_dir)

    rr.init()
    try:
        scene = rr.create_scene()
        scene.load_scene(str(args.scene))
        scene.set_camera(
            (2.5, 2.0, 2.5),
            (0.0, 0.5, 0.0),
            (0.0, 1.0, 0.0),
            45.0,
            args.width,
            args.height,
            0.1,
            100.0,
        )
        rgba = scene.render_frame()
        output_path = args.output or frame_output_path(args.output_dir, args.output_stem, 0)
        write_rgba_bytes_ppm(output_path, rgba, scene.width, scene.height)
    finally:
        rr.destroy()

    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
