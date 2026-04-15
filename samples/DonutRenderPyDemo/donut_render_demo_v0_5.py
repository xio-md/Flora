from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from python_demo_common import default_output_dir, frame_output_path, write_json, write_rgba_bytes_ppm


def _make_box(size: tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sx, sy, sz = (0.5 * float(v) for v in size)
    face_vertices = [
        [(-sx, -sy, -sz), (-sx, sy, -sz), (sx, sy, -sz), (sx, -sy, -sz)],
        [(-sx, -sy, sz), (sx, -sy, sz), (sx, sy, sz), (-sx, sy, sz)],
        [(-sx, -sy, -sz), (sx, -sy, -sz), (sx, -sy, sz), (-sx, -sy, sz)],
        [(sx, -sy, -sz), (sx, sy, -sz), (sx, sy, sz), (sx, -sy, sz)],
        [(sx, sy, -sz), (-sx, sy, -sz), (-sx, sy, sz), (sx, sy, sz)],
        [(-sx, sy, -sz), (-sx, -sy, -sz), (-sx, -sy, sz), (-sx, sy, sz)],
    ]
    face_normals = [
        (0.0, 0.0, -1.0),
        (0.0, 0.0, 1.0),
        (0.0, -1.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (-1.0, 0.0, 0.0),
    ]
    vertices = np.array([vertex for face in face_vertices for vertex in face], dtype=np.float32)
    normals = np.array([normal for normal in face_normals for _ in range(4)], dtype=np.float32)
    triangles = np.array(
        [
            (0, 1, 2),
            (0, 2, 3),
            (4, 5, 6),
            (4, 6, 7),
            (8, 9, 10),
            (8, 10, 11),
            (12, 13, 14),
            (12, 14, 15),
            (16, 17, 18),
            (16, 18, 19),
            (20, 21, 22),
            (20, 22, 23),
        ],
        dtype=np.uint32,
    )
    return vertices, triangles, normals


def _make_plane(size: float) -> tuple[np.ndarray, np.ndarray]:
    half = 0.5 * float(size)
    vertices = np.array(
        [
            (-half, 0.0, -half),
            (half, 0.0, -half),
            (half, 0.0, half),
            (-half, 0.0, half),
        ],
        dtype=np.float32,
    )
    triangles = np.array([(0, 2, 1), (0, 3, 2)], dtype=np.uint32)
    return vertices, triangles


def _rotation_y(angle_degrees: float) -> np.ndarray:
    angle = math.radians(angle_degrees)
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array(
        [
            [c, 0.0, s, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-s, 0.0, c, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def _normalize(vector: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(vector))
    if length <= 1.0e-8:
        raise ValueError("Cannot normalize a zero-length vector.")
    return vector.astype(np.float32) / length


def _camera_pose(position: tuple[float, float, float], target: tuple[float, float, float], up: tuple[float, float, float]) -> np.ndarray:
    pos = np.asarray(position, dtype=np.float32)
    tgt = np.asarray(target, dtype=np.float32)
    up_vector = np.asarray(up, dtype=np.float32)

    forward = _normalize(tgt - pos)
    right = _normalize(np.cross(forward, up_vector))
    camera_up = _normalize(np.cross(right, forward))

    pose = np.eye(4, dtype=np.float32)
    pose[:3, 0] = right
    pose[:3, 1] = camera_up
    pose[:3, 2] = -forward
    pose[:3, 3] = pos
    return pose


def _serialize_plan(plan: dict[str, object]) -> dict[str, object]:
    return {
        "mode": str(plan["mode"]),
        "time": float(plan["time"]),
        "dirty_categories": list(plan["dirty_categories"]),
        "dirty_sources": {key: list(value) for key, value in plan["dirty_sources"].items()},
        "force_render": bool(plan["force_render"]),
        "backend_rebuilt": bool(plan["backend_rebuilt"]),
        "environment_applied": bool(plan["environment_applied"]),
        "operations": [dict(operation) for operation in plan["operations"]],
        "blockers": list(plan["blockers"]),
        "cxx_candidates": list(plan["cxx_candidates"]),
    }


def _environment_emission(frame_index: int) -> tuple[float, float, float]:
    phase = float(frame_index)
    return (
        0.08 + 0.03 * math.sin(phase * 0.45),
        0.10 + 0.04 * math.cos(phase * 0.35),
        0.14 + 0.05 * math.sin(phase * 0.25 + 0.4),
    )


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "python"))

    import DonutRenderPy as dr

    parser = argparse.ArgumentParser(description="Run the Month 2 DonutRenderPy Demo v0.5 with incremental camera/environment/transform updates.")
    parser.add_argument("--module-dir", type=Path, default=repo_root / "bin" / "windows-x64")
    parser.add_argument("--runtime-dir", type=Path, default=repo_root)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir(repo_root, "donut_render_demo_v0_5"))
    parser.add_argument("--output-stem", type=str, default="demo_v05_frame")
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args()

    frame_count = max(1, int(args.frames))
    width = max(1, int(args.width))
    height = max(1, int(args.height))
    output_dir = args.output_dir
    manifest_path = args.manifest or (output_dir / "manifest.json")

    ground_vertices, ground_triangles = _make_plane(7.0)
    box_vertices, box_triangles, box_normals = _make_box((1.2, 1.2, 1.2))

    dr.init(
        context_path=repo_root,
        runtime_dir=args.runtime_dir,
        module_dir=args.module_dir,
        backend="vulkan",
        device_index=-1,
        log_level=dr.LogLevel.WARNING,
    )

    scene = None
    frame_entries: list[dict[str, object]] = []
    try:
        scene = dr.create_scene()
        scene.init(
            dr.Render(
                name="month2-demo-v0.5",
                spectrum=dr.SRGBSpectrum(),
                integrator=dr.WavePathIntegrator(log_level=dr.LogLevel.WARNING, max_depth=8),
                clamp_normal=45.0,
            )
        )

        ground_surface = dr.PlasticSurface(
            name="ground_surface",
            kd=dr.ColorTexture((0.70, 0.72, 0.76, 1.0)),
            roughness=dr.ColorTexture((0.94,)),
        )
        box_surface = dr.PlasticSurface(
            name="box_surface",
            kd=dr.ColorTexture((0.90, 0.50, 0.18, 1.0)),
            roughness=dr.ColorTexture((0.58,)),
        )
        box_light = dr.Light(
            name="box_light",
            emission=dr.ColorTexture((0.18, 0.08, 0.03)),
            intensity=2.6,
        )

        scene.update_environment(
            dr.Environment(
                name="sky",
                emission=dr.ColorTexture(_environment_emission(0)),
            )
        )
        scene.update_surface(ground_surface)
        scene.update_surface(box_surface)
        scene.update_emission(box_light)

        ground = dr.RigidShape(
            name="ground",
            vertices=ground_vertices,
            triangles=ground_triangles,
            surface=ground_surface,
        )
        box = dr.RigidShape(
            name="box",
            vertices=box_vertices,
            triangles=box_triangles,
            normals=box_normals,
            transform=dr.MatrixTransform(np.eye(4, dtype=np.float32)),
            surface=box_surface,
            emission=box_light,
        )
        camera = dr.PinholeCamera(
            name="main_camera",
            pose=dr.MatrixTransform(
                _camera_pose(
                    position=(2.9, 1.8, 3.1),
                    target=(0.0, 0.58, 0.0),
                    up=(0.0, 1.0, 0.0),
                )
            ),
            film=dr.Film((width, height)),
            filter=dr.Filter(radius=1.0),
            spp=4,
            fov=45.0,
        )

        scene.update_shape(ground)
        scene.update_shape(box)
        scene.update_camera(camera, denoise=False)

        for frame_index in range(frame_count):
            time_value = float(frame_index)
            if frame_index > 0:
                scene.update_environment(
                    dr.Environment(
                        name="sky",
                        emission=dr.ColorTexture(_environment_emission(frame_index)),
                    )
                )

                box_transform = _rotation_y(12.0 + frame_index * 18.0)
                box_transform[:3, 3] = np.array(
                    [
                        0.22 * math.sin(frame_index * 0.35),
                        0.60 + 0.06 * math.sin(frame_index * 0.5),
                        0.18 * math.cos(frame_index * 0.35),
                    ],
                    dtype=np.float32,
                )
                box.update(transform=dr.MatrixTransform(box_transform))
                scene.update_shape(box)

                orbit_angle = math.radians(18.0 + frame_index * 7.5)
                camera_pose = _camera_pose(
                    position=(
                        2.9 + 0.25 * math.cos(orbit_angle),
                        1.8 + 0.05 * math.sin(frame_index * 0.3),
                        3.1 + 0.25 * math.sin(orbit_angle),
                    ),
                    target=(0.0, 0.60, 0.0),
                    up=(0.0, 1.0, 0.0),
                )
                camera.update(pose=dr.MatrixTransform(camera_pose))
                scene.update_camera(camera, denoise=False)

            preview = _serialize_plan(scene.preview_update_plan(time=time_value))

            started = time.perf_counter()
            scene.update_scene(time=time_value)
            update_scene_ms = (time.perf_counter() - started) * 1000.0
            report = scene.get_update_stats()

            started = time.perf_counter()
            rgba = scene.render_frame(camera)
            render_frame_ms = (time.perf_counter() - started) * 1000.0

            output_path = frame_output_path(output_dir, args.output_stem, frame_index)
            write_rgba_bytes_ppm(output_path, rgba, width, height)

            report_plan = _serialize_plan(report["plan"])
            frame_entries.append(
                {
                    "frame_index": frame_index,
                    "time": time_value,
                    "path": str(output_path),
                    "preview_mode": preview["mode"],
                    "report_mode": str(report["mode"]),
                    "backend_rebuilt": bool(report["backend_rebuilt"]),
                    "dirty_categories": list(report_plan["dirty_categories"]),
                    "operations": [str(operation["name"]) for operation in report_plan["operations"]],
                    "update_scene_ms": float(update_scene_ms),
                    "reported_update_scene_ms": float(report["duration_ms"]),
                    "render_frame_ms": float(render_frame_ms),
                    "plan": report_plan,
                }
            )

        incremental_frames = sum(1 for entry in frame_entries if str(entry["report_mode"]).startswith("incremental"))
        write_json(
            manifest_path,
            {
                "demo": "DonutRenderPy Demo v0.5",
                "phase": "month2",
                "frame_count": frame_count,
                "incremental_frame_count": incremental_frames,
                "full_rebuild_frame_count": frame_count - incremental_frames,
                "frames": frame_entries,
                "module": "DonutRenderPy",
                "notes": [
                    "Month 2 demo keeps geometry and material topology stable after bootstrap.",
                    "Frames after the initial build combine environment, camera, and rigid transform updates.",
                    "Week 8 native SceneGraph transform updates make the per-frame path Luisa-style and incrementally executable.",
                ],
                "output_dir": str(output_dir),
                "render_flow": [
                    "init",
                    "create_scene",
                    "scene.init(render)",
                    "bootstrap: update_environment/update_surface/update_emission/update_shape/update_camera",
                    "per_frame: update_environment/update_shape/update_camera/preview_update_plan/update_scene/render_frame",
                ],
                "resolution": [width, height],
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
