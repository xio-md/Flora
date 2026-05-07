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
    parser = argparse.ArgumentParser(description="Short regression: deformable + particles + DonutRenderPy.")
    parser.add_argument("--module-dir", type=Path, default=_default_module_dir(repo_root))
    parser.add_argument("--runtime-dir", type=Path, default=repo_root)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, str(repo_root / "python"))
    import DonutRenderPy as dr

    ground_verts = np.array([(-2.0, 0.0, -2.0), (2.0, 0.0, -2.0), (2.0, 0.0, 2.0), (-2.0, 0.0, 2.0)], dtype=np.float32)
    ground_tris = np.array([(0, 2, 1), (0, 3, 2)], dtype=np.uint32)

    sheet_verts = np.array(
        [
            (-0.5, 0.1, -0.5),
            (0.5, 0.1, -0.5),
            (0.5, 0.1, 0.5),
            (-0.5, 0.1, 0.5),
        ],
        dtype=np.float32,
    )
    sheet_tris = np.array([(0, 2, 1), (0, 3, 2)], dtype=np.uint32)

    dr.init(
        context_path=repo_root,
        runtime_dir=args.runtime_dir,
        module_dir=args.module_dir,
        backend="vulkan",
        device_index=-1,
        log_level=dr.LogLevel.WARNING,
    )
    scene = None
    try:
        scene = dr.create_scene()
        scene.init(dr.Render(name="week10-smoke", spectrum=dr.SRGBSpectrum(), integrator=dr.WavePathIntegrator()))
        scene.update_environment(dr.Environment(name="env", emission=dr.ColorTexture((0.07, 0.08, 0.1))))

        sg = dr.PlasticSurface(name="g", kd=dr.ColorTexture((0.5, 0.52, 0.55, 1.0)), roughness=dr.ColorTexture((0.9,)))
        ss = dr.PlasticSurface(name="s", kd=dr.ColorTexture((0.8, 0.4, 0.2, 1.0)), roughness=dr.ColorTexture((0.5,)))
        sp = dr.PlasticSurface(name="p", kd=dr.ColorTexture((0.9, 0.95, 1.0, 1.0)), roughness=dr.ColorTexture((0.15,)))
        scene.update_surface(sg)
        scene.update_surface(ss)
        scene.update_surface(sp)

        ground = dr.RigidShape(name="ground", vertices=ground_verts, triangles=ground_tris, surface=sg)
        sheet = dr.DeformableShape(name="sheet", vertices=sheet_verts, triangles=sheet_tris, surface=ss)
        pts = dr.ParticlesShape(
            name="pts",
            centers=[(0.2, 0.25, 0.1), (-0.15, 0.22, -0.05)],
            radii=[0.06, 0.05],
            surface=sp,
        )
        cam = dr.PinholeCamera(
            name="cam",
            pose=dr.MatrixTransform(
                (
                    (1.0, 0.0, 0.0, 0.0),
                    (0.0, 1.0, 0.0, 1.2),
                    (0.0, 0.0, 1.0, 2.0),
                    (0.0, 0.0, 0.0, 1.0),
                )
            ),
            film=dr.Film((64, 64)),
            filter=dr.Filter(1.0),
            spp=1,
            fov=45.0,
        )
        scene.update_shape(ground)
        scene.update_shape(sheet)
        scene.update_shape(pts)
        scene.update_camera(cam, denoise=False)

        for i in range(5):
            w = sheet_verts.copy()
            w[:, 1] += 0.03 * float(i)
            sheet.update(w, sheet_tris)
            pts.update(
                [
                    (0.2 + 0.02 * i, 0.25, 0.1),
                    (-0.15, 0.22 + 0.01 * i, -0.05),
                ],
                [0.055 + 0.002 * i, 0.048],
            )
            scene.update_shape(sheet)
            scene.update_shape(pts)
            scene.update_scene(time=float(i))
            rgba = scene.render_frame(cam)
            if len(rgba) != 64 * 64 * 4:
                raise AssertionError(f"bad rgba len {len(rgba)}")

        scene.destroy()
        scene = None
    finally:
        if scene is not None:
            scene.destroy()
        dr.destroy()

    if not args.quiet:
        print("week10_dynamic_geometry_smoke passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
