from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from python_demo_common import default_output_dir, frame_output_path, write_json, write_rgba_bytes_ppm


def _checkerboard_rgba() -> bytes:
    pixels = np.array(
        [
            [[255, 110, 40, 255], [30, 180, 255, 255]],
            [[255, 240, 80, 255], [40, 40, 40, 255]],
        ],
        dtype=np.uint8,
    )
    return pixels.tobytes()


def _scalar_texture(values: list[list[int]]) -> bytes:
    return np.asarray(values, dtype=np.uint8).reshape(2, 2, 1).tobytes()


def _emissive_texture() -> bytes:
    pixels = np.array(
        [
            [[255, 96, 32], [32, 32, 32]],
            [[32, 32, 32], [32, 160, 255]],
        ],
        dtype=np.uint8,
    )
    return pixels.tobytes()


def _default_module_dir(repo_root: Path) -> Path:
    platform_dir = "windows-x64" if os.name == "nt" else "linux-x64"
    return repo_root / "bin" / platform_dir


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "python"))

    import DonutRenderPy as dr

    parser = argparse.ArgumentParser(description="Render a textured Week 9 material demo.")
    parser.add_argument("--module-dir", type=Path, default=_default_module_dir(repo_root))
    parser.add_argument("--runtime-dir", type=Path, default=repo_root)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir(repo_root, "donut_render_material_texture_demo"),
    )
    parser.add_argument("--output-stem", type=str, default="material_texture_demo")
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args()

    width = max(1, int(args.width))
    height = max(1, int(args.height))
    output_dir = args.output_dir
    manifest_path = args.manifest or (output_dir / "manifest.json")

    vertices = (
        (-1.2, -1.2, 0.0),
        (1.2, -1.2, 0.0),
        (1.2, 1.2, 0.0),
        (-1.2, 1.2, 0.0),
    )
    triangles = ((0, 1, 2), (0, 2, 3))
    uvs = ((0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0))
    camera_pose = dr.MatrixTransform(
        (
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 3.0),
            (0.0, 0.0, 0.0, 1.0),
        )
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
                name="week9-material-texture-demo",
                spectrum=dr.SRGBSpectrum(),
                integrator=dr.WavePathIntegrator(log_level=dr.LogLevel.WARNING, max_depth=8),
            )
        )
        scene.update_environment(
            dr.Environment(
                name="studio_env",
                emission=dr.ColorTexture((0.08, 0.09, 0.10)),
            )
        )

        surface = dr.DisneySurface(
            name="textured_surface",
            kd=dr.ImageTexture(
                image_data=_checkerboard_rgba(),
                width=2,
                height=2,
                channel=4,
                encoding="raw",
            ),
            roughness=dr.ImageTexture(
                image_data=_scalar_texture([[32, 128], [200, 255]]),
                width=2,
                height=2,
                channel=1,
                encoding="raw",
            ),
            metallic=dr.ImageTexture(
                image_data=_scalar_texture([[255, 16], [48, 220]]),
                width=2,
                height=2,
                channel=1,
                encoding="raw",
            ),
            opacity=dr.ImageTexture(
                image_data=_scalar_texture([[255, 255], [80, 255]]),
                width=2,
                height=2,
                channel=1,
                encoding="raw",
            ),
            double_sided=True,
        )
        emission = dr.Light(
            name="textured_emission",
            emission=dr.ImageTexture(
                image_data=_emissive_texture(),
                width=2,
                height=2,
                channel=3,
                encoding="raw",
            ),
            intensity=1.5,
        )
        shape = dr.RigidShape(
            name="textured_quad",
            vertices=vertices,
            triangles=triangles,
            uvs=uvs,
            surface=surface,
            emission=emission,
        )
        camera = dr.PinholeCamera(
            name="main_camera",
            pose=camera_pose,
            film=dr.Film((width, height)),
            filter=dr.Filter(1.0),
            spp=4,
            fov=40.0,
        )

        scene.update_surface(surface)
        scene.update_emission(emission)
        scene.update_shape(shape)
        scene.update_camera(camera, denoise=False)
        rgba = scene.render_frame(camera)

        output_path = frame_output_path(output_dir, args.output_stem, 0)
        write_rgba_bytes_ppm(output_path, rgba, width, height)
        write_json(
            manifest_path,
            {
                "demo": "DonutRenderPy Material Texture Demo",
                "frame_count": 1,
                "module": "DonutRenderPy",
                "output": str(output_path),
                "resolution": [width, height],
                "supported_features": [
                    "base_color_texture",
                    "metallic_texture",
                    "roughness_texture",
                    "opacity_texture",
                    "emissive_texture",
                ],
            },
        )
    finally:
        if scene is not None:
            scene.destroy()
        dr.destroy()

    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
