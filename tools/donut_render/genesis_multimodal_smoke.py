"""Validate GenesisStyleRenderer dynamic multimodal sensor calls."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np


def default_module_dir(repo_root: Path) -> Path:
    platform_dir = "windows-x64" if os.name == "nt" else "linux-x64"
    return repo_root / "bin" / platform_dir


def make_box(size: float) -> tuple[np.ndarray, np.ndarray]:
    half = 0.5 * size
    faces = [
        [(-half, -half, -half), (half, -half, -half), (half, half, -half), (-half, half, -half)],
        [(-half, -half, half), (-half, half, half), (half, half, half), (half, -half, half)],
        [(-half, -half, -half), (-half, -half, half), (half, -half, half), (half, -half, -half)],
        [(half, -half, -half), (half, -half, half), (half, half, half), (half, half, -half)],
        [(-half, half, -half), (half, half, -half), (half, half, half), (-half, half, half)],
        [(-half, -half, -half), (-half, half, -half), (-half, half, half), (-half, -half, half)],
    ]
    vertices = np.asarray([vertex for face in faces for vertex in face], dtype=np.float32)
    local_triangles = np.asarray(((0, 2, 1), (0, 3, 2)), dtype=np.uint32)
    triangles = np.vstack(
        [local_triangles + face_index * 4 for face_index in range(6)]
    )
    return vertices, triangles


def validate_frame(frame, resolution: tuple[int, int]) -> None:
    width, height = resolution
    expected = {
        "color": ((height, width, 4), np.dtype(np.uint8)),
        "depth": ((height, width), np.dtype(np.float32)),
        "normal": ((height, width, 3), np.dtype(np.float32)),
        "instance": ((height, width), np.dtype(np.uint32)),
        "semantic": ((height, width), np.dtype(np.uint32)),
    }
    for product, (shape, dtype) in expected.items():
        array = getattr(frame, product)
        if array is None or array.shape != shape or array.dtype != dtype:
            raise AssertionError(
                f"Invalid {product}: {None if array is None else (array.shape, array.dtype)}"
            )
    depth_valid = frame.depth > 0.0
    normal_valid = np.linalg.norm(frame.normal, axis=2) > 0.5
    instance_valid = frame.instance > 0
    if not np.array_equal(depth_valid, normal_valid):
        raise AssertionError("Depth and normal masks are not aligned.")
    if not np.array_equal(depth_valid, instance_valid):
        raise AssertionError("Depth and instance masks are not aligned.")
    if not set(np.unique(frame.instance)).issubset({0, 1, 2}):
        raise AssertionError("Genesis instance labels are not stable sequential IDs.")
    if np.any(frame.semantic != 0):
        raise AssertionError("Unlabeled Genesis geometry must use semantic ID 0.")


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module-dir", type=Path, default=default_module_dir(repo_root))
    parser.add_argument("--runtime-dir", type=Path, default=repo_root)
    args = parser.parse_args()

    sys.path.insert(0, str(repo_root / "python"))
    from rtxns_genesis_style import CameraDesc, GenesisStyleRenderer, SurfaceDesc

    ground_vertices = np.asarray(
        ((-2.0, 0.0, -2.0), (2.0, 0.0, -2.0), (2.0, 0.0, 2.0), (-2.0, 0.0, 2.0)),
        dtype=np.float32,
    )
    ground_triangles = np.asarray(((0, 2, 1), (0, 3, 2)), dtype=np.uint32)
    box_vertices, box_triangles = make_box(0.7)
    camera0 = CameraDesc(
        uid="cam0",
        pos=(1.8, 1.5, 2.5),
        lookat=(0.0, 0.3, 0.0),
        res=(96, 72),
        fov=50.0,
        near=0.05,
        far=20.0,
    )
    camera1 = CameraDesc(
        uid="cam1",
        pos=(-1.8, 1.2, 2.0),
        lookat=(0.0, 0.3, 0.0),
        res=(96, 72),
        fov=50.0,
        near=0.05,
        far=20.0,
    )

    with GenesisStyleRenderer(
        module_dir=args.module_dir,
        runtime_dir=args.runtime_dir,
    ) as renderer:
        renderer.set_ambient((0.08, 0.08, 0.08), (0.03, 0.03, 0.03))
        renderer.set_default_light((-0.5, -1.0, -0.4), irradiance=1.5)
        renderer.add_surface(
            "ground", SurfaceDesc(base_color=(0.55, 0.58, 0.62, 1.0), roughness=0.9)
        )
        renderer.add_surface(
            "box", SurfaceDesc(base_color=(0.85, 0.25, 0.12, 1.0), roughness=0.45)
        )
        renderer.add_rigid("ground", ground_vertices, ground_triangles)
        renderer.add_rigid("box", box_vertices, box_triangles)
        box_pose = np.eye(4, dtype=np.float32)
        box_pose[:3, 3] = (0.0, 0.35, 0.0)
        renderer.update_rigid("box", box_pose)
        renderer.add_camera(camera0)
        renderer.add_camera(camera1)

        before = renderer.render_sensor_batch((camera0, camera1))
        for frame in before:
            validate_frame(frame, camera0.res)
        initial_camera_count = int(renderer._scene.camera_count)

        for legacy_camera in (camera1, camera0, camera1, camera0):
            renderer.render_camera(legacy_camera)
            interleaved = renderer.render_sensor_batch((camera0, camera1))
            for frame in interleaved:
                validate_frame(frame, camera0.res)
        if int(renderer._scene.camera_count) != initial_camera_count:
            raise AssertionError("Interleaved legacy/sensor calls leaked camera slots.")

        box_pose[0, 3] = 0.45
        renderer.update_rigid("box", box_pose)
        after = renderer.render_sensor(camera0)
        validate_frame(after, camera0.res)
        if np.array_equal(before[0].instance, after.instance):
            raise AssertionError("Rigid pose update did not change the instance image.")

        partial = renderer.render_sensor(camera0, products=("depth", "instance"))
        if partial.products() != ("depth", "instance"):
            raise AssertionError(f"Unexpected partial products: {partial.products()}")

    quad_vertices = np.asarray(
        ((-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (1.0, 1.0, 0.0), (-1.0, 1.0, 0.0)),
        dtype=np.float32,
    )
    quad_triangles = np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.uint32)
    depth_camera = CameraDesc(
        uid="depth_reference",
        pos=(0.0, 0.0, 3.0),
        lookat=(0.0, 0.0, 0.0),
        res=(65, 65),
        fov=45.0,
        near=0.1,
        far=10.0,
    )
    with GenesisStyleRenderer(
        module_dir=args.module_dir,
        runtime_dir=args.runtime_dir,
    ) as renderer:
        renderer.add_surface(
            "reference_quad",
            SurfaceDesc(
                base_color=(0.8, 0.8, 0.8, 1.0),
                roughness=1.0,
                double_sided=True,
            ),
        )
        renderer.add_rigid("reference_quad", quad_vertices, quad_triangles)
        renderer.add_camera(depth_camera)
        depth_frame = renderer.render_sensor(
            depth_camera, products=("depth", "normal", "instance")
        )
        center_depth = float(depth_frame.depth[32, 32])
        depth_error = abs(center_depth - 3.0)
        if depth_error > 1.0e-4:
            raise AssertionError(
                f"Linear depth reference error is {depth_error}; expected <= 1e-4."
            )

    print(
        "PASS: GenesisStyleRenderer returned aligned five-product frames for two "
        "cameras and preserved the API across a rigid pose update; "
        f"depth reference error={depth_error:.3e} m.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
