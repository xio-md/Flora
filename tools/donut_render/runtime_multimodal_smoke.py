"""Validate the public donut_render_py.Scene multimodal sensor wrapper."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np


def default_module_dir(repo_root: Path) -> Path:
    platform_dir = "windows-x64" if os.name == "nt" else "linux-x64"
    return repo_root / "bin" / platform_dir


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module-dir", type=Path, default=default_module_dir(repo_root))
    parser.add_argument("--runtime-dir", type=Path, default=repo_root)
    args = parser.parse_args()
    sys.path.insert(0, str(repo_root / "python"))

    import donut_render_py as dr

    vertices = (
        (-1.0, -1.0, 0.0),
        (1.0, -1.0, 0.0),
        (1.0, 1.0, 0.0),
        (-1.0, 1.0, 0.0),
    )
    triangles = ((0, 1, 2), (0, 2, 3))
    camera = dr.PinholeCamera(
        name="sensor_camera",
        pose=dr.MatrixTransform(
            (
                (1.0, 0.0, 0.0, 0.0),
                (0.0, 1.0, 0.0, 0.0),
                (0.0, 0.0, 1.0, 3.0),
                (0.0, 0.0, 0.0, 1.0),
            )
        ),
        film=dr.Film((64, 48)),
        filter=dr.Filter(1.0),
        spp=1,
        fov=45.0,
    )
    surface = dr.PlasticSurface(
        name="sensor_surface",
        kd=dr.ColorTexture((0.8, 0.3, 0.2, 1.0)),
        roughness=dr.ColorTexture((0.7,)),
        double_sided=True,
    )
    shape = dr.RigidShape(
        name="sensor_quad",
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
    scene = None
    try:
        scene = dr.create_scene()
        scene.init(
            dr.Render(
                name="runtime-multimodal-smoke",
                spectrum=dr.SRGBSpectrum(),
                integrator=dr.WavePathIntegrator(),
            )
        )
        scene.update_surface(surface)
        scene.update_shape(shape)
        scene.update_camera(camera, denoise=False)
        frame = scene.render_sensor(camera)
        expected_shapes = {
            "color": (48, 64, 4),
            "depth": (48, 64),
            "normal": (48, 64, 3),
            "instance": (48, 64),
            "semantic": (48, 64),
        }
        for product, shape_value in expected_shapes.items():
            array = getattr(frame, product)
            if array is None or array.shape != shape_value:
                raise AssertionError(f"Invalid runtime {product} shape.")
        if abs(float(frame.depth[24, 32]) - 3.0) > 1.0e-4:
            raise AssertionError("Public Scene wrapper returned incorrect linear depth.")
        batch = scene.render_sensor_batch((camera,), products=("depth", "instance"))
        if len(batch) != 1 or batch[0].products() != ("depth", "instance"):
            raise AssertionError("Public Scene partial-product batch contract failed.")
    finally:
        if scene is not None:
            scene.destroy()
        dr.destroy()

    print("PASS: donut_render_py.Scene multimodal sensor wrapper.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
