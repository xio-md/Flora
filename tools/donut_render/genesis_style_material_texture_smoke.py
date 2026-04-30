from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np


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
    parser = argparse.ArgumentParser(description="Validate GenesisStyleRenderer material texture support.")
    parser.add_argument("--module-dir", type=Path, default=_default_module_dir(repo_root))
    parser.add_argument("--runtime-dir", type=Path, default=repo_root)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, str(repo_root / "python"))

    from rtxns_genesis_style import CameraDesc, GenesisStyleRenderer

    vertices = np.array(
        [
            (-1.0, -1.0, 0.0),
            (1.0, -1.0, 0.0),
            (1.0, 1.0, 0.0),
            (-1.0, 1.0, 0.0),
        ],
        dtype=np.float32,
    )
    triangles = np.array([(0, 1, 2), (0, 2, 3)], dtype=np.uint32)
    uvs = np.array(
        [
            (0.0, 1.0),
            (1.0, 1.0),
            (1.0, 0.0),
            (0.0, 0.0),
        ],
        dtype=np.float32,
    )

    surface = {
        "name": "quad_surface",
        "base_color": {
            "image_data": bytes(
                [
                    255,
                    64,
                    32,
                    255,
                    32,
                    180,
                    255,
                    255,
                    255,
                    240,
                    96,
                    255,
                    48,
                    48,
                    48,
                    128,
                ]
            ),
            "width": 2,
            "height": 2,
            "channel": 4,
            "encoding": "raw",
        },
        "roughness": {
            "image_data": bytes([24, 128, 200, 255]),
            "width": 2,
            "height": 2,
            "channel": 1,
            "encoding": "raw",
        },
        "metallic": {
            "image_data": bytes([255, 0, 80, 220]),
            "width": 2,
            "height": 2,
            "channel": 1,
            "encoding": "raw",
        },
        "emissive": {
            "image_data": bytes(
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
            ),
            "width": 2,
            "height": 2,
            "channel": 3,
            "encoding": "raw",
        },
        "opacity": {
            "image_data": bytes([255, 255, 96, 255]),
            "width": 2,
            "height": 2,
            "channel": 1,
            "encoding": "raw",
        },
        "double_sided": True,
    }
    camera = CameraDesc(
        uid="cam",
        pos=(0.0, 0.0, 3.0),
        lookat=(0.0, 0.0, 0.0),
        up=(0.0, 1.0, 0.0),
        res=(64, 64),
        fov=45.0,
        near=0.1,
        far=100.0,
    )

    with GenesisStyleRenderer(module_dir=args.module_dir, runtime_dir=args.runtime_dir) as renderer:
        renderer.add_surface("quad", surface)
        renderer.add_rigid("quad", vertices, triangles, uvs=uvs)
        renderer.add_camera(camera)
        image = renderer.render_camera(camera, force_render=True, time=0.0)
        if image.shape != (64, 64, 3):
            raise AssertionError(f"Unexpected image shape: {image.shape}")
        document = _read_glb_json(Path(renderer._scene_path))

    materials = list(document.get("materials", []))
    if len(materials) != 1:
        raise AssertionError(f"Expected one material, found {len(materials)}")
    material = materials[0]
    pbr = material.get("pbrMetallicRoughness", {})
    if "baseColorTexture" not in pbr:
        raise AssertionError("baseColorTexture was not emitted for GenesisStyleRenderer.")
    if "metallicRoughnessTexture" not in pbr:
        raise AssertionError("metallicRoughnessTexture was not emitted for GenesisStyleRenderer.")
    if "emissiveTexture" not in material:
        raise AssertionError("emissiveTexture was not emitted for GenesisStyleRenderer.")
    if material.get("alphaMode") != "BLEND":
        raise AssertionError(f"Expected alphaMode=BLEND, got {material.get('alphaMode')!r}")
    if len(document.get("images", [])) < 3 or len(document.get("textures", [])) < 3:
        raise AssertionError("Expected embedded textures in GenesisStyleRenderer GLB output.")
    if not args.quiet:
        print("GenesisStyleRenderer material texture smoke passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
