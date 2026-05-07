from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np


def _default_module_dir(repo_root: Path) -> Path:
    platform_dir = "windows-x64" if os.name == "nt" else "linux-x64"
    return repo_root / "bin" / platform_dir


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="GenesisStyleRenderer deformable + particles short regression.")
    parser.add_argument("--module-dir", type=Path, default=_default_module_dir(repo_root))
    parser.add_argument("--runtime-dir", type=Path, default=repo_root)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, str(repo_root / "python"))
    from rtxns_genesis_style import CameraDesc, GenesisStyleRenderer, SurfaceDesc

    ground_verts = np.array([(-2.0, 0.0, -2.0), (2.0, 0.0, -2.0), (2.0, 0.0, 2.0), (-2.0, 0.0, 2.0)], dtype=np.float32)
    ground_tris = np.array([(0, 2, 1), (0, 3, 2)], dtype=np.uint32)
    sheet_verts = np.array(
        [(-0.4, 0.12, -0.4), (0.4, 0.12, -0.4), (0.4, 0.12, 0.4), (-0.4, 0.12, 0.4)],
        dtype=np.float32,
    )
    sheet_tris = np.array([(0, 2, 1), (0, 3, 2)], dtype=np.uint32)

    cam = CameraDesc(
        uid="cam",
        pos=(0.0, 1.1, 2.2),
        lookat=(0.0, 0.15, 0.0),
        up=(0.0, 1.0, 0.0),
        res=(48, 48),
        fov=48.0,
        near=0.05,
        far=80.0,
    )

    with GenesisStyleRenderer(module_dir=args.module_dir, runtime_dir=args.runtime_dir) as renderer:
        renderer.set_ambient((0.06, 0.07, 0.08), (0.03, 0.03, 0.04))
        renderer.set_default_light(direction=(-0.5, -1.0, -0.4), color=(1.0, 0.95, 0.9), irradiance=0.85)

        renderer.add_surface("g", SurfaceDesc(base_color=(0.5, 0.52, 0.55, 1.0), roughness=0.9))
        renderer.add_surface("s", SurfaceDesc(base_color=(0.78, 0.4, 0.2, 1.0), roughness=0.45))
        renderer.add_surface("p", SurfaceDesc(base_color=(0.9, 0.94, 1.0, 1.0), roughness=0.18))

        renderer.add_rigid("ground", ground_verts, ground_tris)
        renderer.add_deformable("sheet")
        renderer.update_deformable("sheet", sheet_verts, sheet_tris)
        renderer.add_particles("pts", radius=0.05)
        renderer.update_particles(
            "pts",
            [(0.15, 0.2, 0.05), (-0.1, 0.18, -0.08)],
            particles_radii=[0.055, 0.048],
        )
        renderer.add_camera(cam)

        for i in range(10):
            w = sheet_verts.copy()
            w[:, 1] += 0.02 * float(i)
            renderer.update_deformable("sheet", w, sheet_tris)
            renderer.update_particles(
                "pts",
                [
                    (0.15 + 0.015 * i, 0.2, 0.05),
                    (-0.1, 0.18 + 0.01 * i, -0.08),
                ],
                particles_radii=[0.052 + 0.001 * i, 0.046],
            )
            img = renderer.render_camera(cam, force_render=True, time=float(i))
            if img.shape != (48, 48, 3):
                raise AssertionError(img.shape)

    if not args.quiet:
        print("week10_genesis_dynamic_smoke passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
