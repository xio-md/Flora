from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _default_module_dir(repo_root: Path) -> Path:
    platform_dir = "windows-x64" if os.name == "nt" else "linux-x64"
    return repo_root / "bin" / platform_dir


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser(description="Run a minimal backend smoke render and print RGBA byte length.")
    parser.add_argument("--module-dir", type=Path, default=_default_module_dir(repo_root))
    parser.add_argument("--runtime-dir", type=Path, default=repo_root)
    args = parser.parse_args()

    sys.path.insert(0, str(repo_root / "python"))

    import DonutRenderPy as dr

    vertices = (
        (-1.0, -1.0, 0.0),
        (1.0, -1.0, 0.0),
        (1.0, 1.0, 0.0),
        (-1.0, 1.0, 0.0),
    )
    triangles = (
        (0, 1, 2),
        (0, 2, 3),
    )

    pose = dr.MatrixTransform(
        (
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 3.0),
            (0.0, 0.0, 0.0, 1.0),
        )
    )

    camera = dr.PinholeCamera(
        name="cam",
        pose=pose,
        film=dr.Film((64, 64)),
        filter=dr.Filter(1.0),
        spp=1,
        fov=45.0,
    )
    surface = dr.PlasticSurface(
        name="mat",
        kd=dr.ColorTexture((0.8, 0.3, 0.2, 1.0)),
        roughness=dr.ColorTexture((0.7,)),
    )
    shape = dr.RigidShape(
        name="quad",
        vertices=vertices,
        triangles=triangles,
        surface=surface,
    )

    dr.init(
        context_path=repo_root,
        runtime_dir=args.runtime_dir,
        module_dir=args.module_dir,
        backend="vulkan",
        device_index=-1,
        log_level=dr.LogLevel.WARNING,
    )
    try:
        scene = dr.create_scene()
        scene.init(
            dr.Render(
                name="backend-smoke",
                spectrum=dr.SRGBSpectrum(),
                integrator=dr.WavePathIntegrator(),
            )
        )
        scene.update_surface(surface)
        scene.update_shape(shape)
        scene.update_camera(camera, denoise=False)
        rgba = scene.render_frame(camera)
        print(len(rgba))
        scene.destroy()
    finally:
        dr.destroy()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
