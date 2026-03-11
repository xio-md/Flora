from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _write_ppm(path: Path, rgba: bytes, width: int, height: int) -> None:
    rgb = bytearray(width * height * 3)
    for src, dst in zip(range(0, len(rgba), 4), range(0, len(rgb), 3)):
        rgb[dst:dst + 3] = rgba[src:src + 3]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
        f.write(rgb)


def main() -> int:
    parser = argparse.ArgumentParser(description="Render one offscreen PBR frame with RtxRenderPy.")
    parser.add_argument("--module-dir", type=Path, default=Path(__file__).resolve().parents[2] / "bin" / "windows-x64")
    parser.add_argument("--scene", type=Path,
                        default=Path(__file__).resolve().parents[2] / "external" / "donut" / "thirdparty" / "cgltf" / "fuzz" / "data" / "Box.glb")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().with_name("headless_pbr_box.ppm"))
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    args = parser.parse_args()

    sys.path.insert(0, str(args.module_dir))

    import RtxRenderPy as rr

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
        _write_ppm(args.output, rgba, scene.width, scene.height)
    finally:
        rr.destroy()

    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
