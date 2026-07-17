#!/usr/bin/env python3
"""Render a ReplicaCAD scene using the Flora Donut native renderer."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    module_dir = repo_root / "bin" / "windows-x64"
    sys.path.insert(0, str(module_dir))

    import DonutRenderPyNative as rr

    scene_path = repo_root / "data" / "replica_cad" / "frl_apartment_stage.glb"
    output_dir = repo_root / "data" / "replica_cad"
    output_dir.mkdir(parents=True, exist_ok=True)

    rr.init()
    try:
        scene = rr.create_scene()
        print(f"[1/3] Loading scene: {scene_path.name}")
        scene.load_scene(str(scene_path))

        # Explicit cross-renderer baseline: Flora's native direct-light setup.
        # It deliberately contains no global-illumination solve.  The SAPIEN
        # --gi render uses the same directional-light direction and an
        # exposure-calibrated equivalent intensity, then adds indirect bounces.
        scene.set_ambient(
            (0.03, 0.04, 0.06),
            (0.01, 0.01, 0.01),
        )
        scene.set_default_light(
            (-0.4, -1.0, -0.6),
            (1.0, 1.0, 1.0),
            2.0,
        )
        # The cross-renderer direct-light baseline includes FloraRT shadows;
        # it intentionally stops short of any global-illumination solve.
        # `--no-rt-shadows` keeps the same RT composite/tonemap path while
        # clearing its visibility mask, for a direct-light control image.
        # Therefore both Flora reference images use the shared .60/HBD
        # display transform implemented in shadow_composite_cs.hlsl.
        rt_shadows_enabled = "--no-rt-shadows" not in sys.argv
        scene.enable_rt_shadows(rt_shadows_enabled)
        scene.enable_shadow_blur("--no-shadow-blur" not in sys.argv)
        scene.set_shadow_samples(8)

        print("[2/3] Setting camera ...")
        scene.set_camera(
            (3.5, 2.0, 3.5),   # position
            (0.0, 1.0, 0.0),   # lookat
            (0.0, 1.0, 0.0),   # up
            60.0,               # fov
            1280,               # width
            720,                # height
            0.1,                # near
            100.0,              # far
        )

        print("[3/3] Rendering ...")
        rgba = scene.render_frame()
        w, h = scene.width, scene.height
        print(f"  Done: {w}x{h}, {len(rgba)} bytes")

        image = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
        rgb = image[:, :, :3]

        # PPM output (always)
        output_stem = "demo_output" if rt_shadows_enabled else "flora_no_rt_shadow_output"
        ppm_path = output_dir / f"{output_stem}.ppm"
        with ppm_path.open("wb") as f:
            f.write(f"P6\n{w} {h}\n255\n".encode("ascii"))
            f.write(np.ascontiguousarray(rgb).tobytes())
        print(f"  Saved PPM: {ppm_path}")

        # PNG output (if Pillow available)
        try:
            from PIL import Image
            png_path = output_dir / f"{output_stem}.png"
            Image.fromarray(rgb, "RGB").save(str(png_path))
            print(f"  Saved PNG: {png_path}")
        except ImportError:
            print("  (Pillow not available — PPM only)")

    finally:
        rr.destroy()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
