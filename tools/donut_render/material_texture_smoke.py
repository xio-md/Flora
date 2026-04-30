from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from pathlib import Path


def _default_module_dir(repo_root: Path) -> Path:
    platform_dir = "windows-x64" if os.name == "nt" else "linux-x64"
    return repo_root / "bin" / platform_dir


def _read_glb_json(path: Path) -> dict[str, object]:
    blob = path.read_bytes()
    if len(blob) < 20:
        raise AssertionError(f"GLB is too small: {path}")
    magic, version, _length = struct.unpack_from("<III", blob, 0)
    if magic != 0x46546C67 or version != 2:
        raise AssertionError(f"Unexpected GLB header: {path}")
    json_length, chunk_type = struct.unpack_from("<II", blob, 12)
    if chunk_type != 0x4E4F534A:
        raise AssertionError("First GLB chunk is not JSON.")
    json_start = 20
    json_end = json_start + json_length
    return json.loads(blob[json_start:json_end].decode("utf-8").rstrip(" "))


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser(description="Validate Week 9 material/texture support.")
    parser.add_argument("--module-dir", type=Path, default=_default_module_dir(repo_root))
    parser.add_argument("--runtime-dir", type=Path, default=repo_root)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, str(repo_root / "python"))

    import DonutRenderPy as dr

    vertices = (
        (-1.0, -1.0, 0.0),
        (1.0, -1.0, 0.0),
        (1.0, 1.0, 0.0),
        (-1.0, 1.0, 0.0),
    )
    triangles = ((0, 1, 2), (0, 2, 3))
    uvs = ((0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0))
    checkerboard = bytes(
        [
            255,
            110,
            40,
            255,
            30,
            180,
            255,
            255,
            255,
            240,
            80,
            255,
            40,
            40,
            40,
            255,
        ]
    )
    roughness = bytes([24, 128, 192, 255])
    metallic = bytes([255, 0, 64, 220])
    opacity = bytes([255, 255, 64, 255])
    emissive = bytes(
        [
            255,
            96,
            32,
            24,
            24,
            24,
            24,
            24,
            24,
            32,
            160,
            255,
        ]
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
                name="material-texture-smoke",
                spectrum=dr.SRGBSpectrum(),
                integrator=dr.WavePathIntegrator(log_level=dr.LogLevel.WARNING),
            )
        )
        surface = dr.DisneySurface(
            name="textured_surface",
            kd=dr.ImageTexture(image_data=checkerboard, width=2, height=2, channel=4, encoding="raw"),
            roughness=dr.ImageTexture(image_data=roughness, width=2, height=2, channel=1, encoding="raw"),
            metallic=dr.ImageTexture(image_data=metallic, width=2, height=2, channel=1, encoding="raw"),
            opacity=dr.ImageTexture(image_data=opacity, width=2, height=2, channel=1, encoding="raw"),
        )
        light = dr.Light(
            name="textured_emission",
            emission=dr.ImageTexture(image_data=emissive, width=2, height=2, channel=3, encoding="raw"),
            intensity=1.25,
        )
        shape = dr.RigidShape(
            name="quad",
            vertices=vertices,
            triangles=triangles,
            uvs=uvs,
            surface=surface,
            emission=light,
        )
        camera = dr.PinholeCamera(
            name="cam",
            pose=dr.MatrixTransform(
                (
                    (1.0, 0.0, 0.0, 0.0),
                    (0.0, 1.0, 0.0, 0.0),
                    (0.0, 0.0, 1.0, 3.0),
                    (0.0, 0.0, 0.0, 1.0),
                )
            ),
            film=dr.Film((64, 64)),
            filter=dr.Filter(1.0),
            spp=1,
            fov=45.0,
        )

        scene.update_environment(dr.Environment(name="env", emission=dr.ColorTexture((0.06, 0.07, 0.08))))
        scene.update_surface(surface)
        scene.update_emission(light)
        scene.update_shape(shape)
        scene.update_camera(camera, denoise=False)
        rgba = scene.render_frame(camera)
        if len(rgba) != 64 * 64 * 4:
            raise AssertionError(f"Unexpected RGBA byte length: {len(rgba)}")

        backend = getattr(scene, "_backend", None)
        scene_glb = None if backend is None else getattr(backend, "_scene_path", None)
        if scene_glb is None:
            raise AssertionError("Scene backend did not expose a scene.glb path for inspection.")
        scene_glb = Path(scene_glb)
        document = _read_glb_json(scene_glb)
        materials = list(document.get("materials", []))
        if len(materials) != 1:
            raise AssertionError(f"Expected one material, found {len(materials)}")
        material = materials[0]
        pbr = material.get("pbrMetallicRoughness", {})
        if "baseColorTexture" not in pbr:
            raise AssertionError("baseColorTexture was not emitted.")
        if "metallicRoughnessTexture" not in pbr:
            raise AssertionError("metallicRoughnessTexture was not emitted.")
        if "emissiveTexture" not in material:
            raise AssertionError("emissiveTexture was not emitted.")
        if material.get("alphaMode") != "BLEND":
            raise AssertionError(f"Expected alphaMode=BLEND, got {material.get('alphaMode')!r}")
        if len(document.get("images", [])) < 3 or len(document.get("textures", [])) < 3:
            raise AssertionError("Expected embedded GLB images/textures for Week 9 material support.")
        if not args.quiet:
            print("Material texture smoke passed.")
    finally:
        if scene is not None:
            scene.destroy()
        dr.destroy()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
