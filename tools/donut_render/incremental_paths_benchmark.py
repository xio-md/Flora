from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def _camera_pose(z_offset: float, x_offset: float = 0.0) -> np.ndarray:
    return np.array(
        (
            (1.0, 0.0, 0.0, x_offset),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, z_offset),
            (0.0, 0.0, 0.0, 1.0),
        ),
        dtype=np.float32,
    )


def _quad_vertices(scale: float = 1.0) -> tuple[tuple[float, float, float], ...]:
    half = float(scale)
    return (
        (-half, -half, 0.0),
        (half, -half, 0.0),
        (half, half, 0.0),
        (-half, half, 0.0),
    )


def _quad_triangles() -> tuple[tuple[int, int, int], ...]:
    return (
        (0, 1, 2),
        (0, 2, 3),
    )


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


def _capture_step(scene, camera, *, step: str, time_value: float) -> dict[str, object]:
    preview = _serialize_plan(scene.preview_update_plan(time=time_value))

    started = time.perf_counter()
    scene.update_scene(time=time_value)
    update_scene_ms = (time.perf_counter() - started) * 1000.0

    report = scene.get_update_stats()

    started = time.perf_counter()
    rgba = scene.render_frame(camera)
    render_frame_ms = (time.perf_counter() - started) * 1000.0

    return {
        "step": step,
        "preview": preview,
        "report_mode": str(report["mode"]),
        "report_plan": _serialize_plan(report["plan"]),
        "update_scene_ms": float(update_scene_ms),
        "reported_update_scene_ms": float(report["duration_ms"]),
        "render_frame_ms": float(render_frame_ms),
        "total_ms": float(update_scene_ms + render_frame_ms),
        "rgba_bytes": len(rgba),
    }


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "python"))

    import DonutRenderPy as dr

    parser = argparse.ArgumentParser(description="Benchmark native incremental update paths.")
    parser.add_argument("--module-dir", type=Path, default=repo_root / "bin" / "windows-x64")
    parser.add_argument("--runtime-dir", type=Path, default=repo_root)
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / ".temp" / "incremental_paths_benchmark.json",
    )
    args = parser.parse_args()

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

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
                name="incremental-paths-benchmark",
                spectrum=dr.SRGBSpectrum(),
                integrator=dr.WavePathIntegrator(log_level=dr.LogLevel.WARNING),
            )
        )

        surface = dr.PlasticSurface(
            name="mat",
            kd=dr.ColorTexture((0.8, 0.3, 0.2, 1.0)),
            roughness=dr.ColorTexture((0.7,)),
        )
        camera = dr.PinholeCamera(
            name="cam",
            pose=dr.MatrixTransform(_camera_pose(3.0)),
            film=dr.Film((64, 64)),
            filter=dr.Filter(1.0),
            spp=1,
            fov=45.0,
        )
        shape = dr.RigidShape(
            name="quad",
            vertices=_quad_vertices(1.0),
            triangles=_quad_triangles(),
            transform=dr.MatrixTransform(np.eye(4, dtype=np.float32)),
            surface="mat",
        )

        scene.update_environment(
            dr.Environment(
                name="env",
                emission=dr.ColorTexture((0.08, 0.10, 0.14)),
            )
        )
        scene.update_surface(surface)
        scene.update_shape(shape)
        scene.update_camera(camera, denoise=False)

        results: list[dict[str, object]] = []
        results.append(_capture_step(scene, camera, step="initial_build", time_value=0.0))

        camera.update(pose=dr.MatrixTransform(_camera_pose(3.2, x_offset=0.15)))
        scene.update_camera(camera, denoise=False)
        results.append(_capture_step(scene, camera, step="camera_only", time_value=1.0))

        scene.update_environment(
            dr.Environment(
                name="env",
                emission=dr.ColorTexture((0.15, 0.18, 0.22)),
            )
        )
        results.append(_capture_step(scene, camera, step="environment_only", time_value=2.0))

        transform = np.eye(4, dtype=np.float32)
        transform[:3, 3] = np.array((0.25, 0.0, 0.0), dtype=np.float32)
        shape.update(transform=dr.MatrixTransform(transform))
        scene.update_shape(shape)
        results.append(_capture_step(scene, camera, step="transform_only", time_value=3.0))

        surface = dr.PlasticSurface(
            name="mat",
            kd=dr.ColorTexture((0.2, 0.6, 0.9, 1.0)),
            roughness=dr.ColorTexture((0.35,)),
        )
        scene.update_surface(surface)
        results.append(_capture_step(scene, camera, step="surface_only", time_value=4.0))

        expected = {
            "initial_build": ("full_rebuild", {"camera", "environment", "geometry", "surface"}, True),
            "camera_only": ("incremental_camera_environment", {"camera"}, False),
            "environment_only": ("incremental_camera_environment", {"environment"}, False),
            "transform_only": ("incremental_camera_transform_environment", {"transform"}, False),
            "surface_only": ("full_rebuild", {"surface"}, True),
        }

        for entry in results:
            expected_mode, expected_categories, expected_rebuild = expected[entry["step"]]
            plan = entry["report_plan"]
            if entry["report_mode"] != expected_mode:
                raise AssertionError(f"{entry['step']} expected mode {expected_mode}, got {entry['report_mode']}.")
            if set(plan["dirty_categories"]) != expected_categories:
                raise AssertionError(
                    f"{entry['step']} expected categories {sorted(expected_categories)}, got {sorted(plan['dirty_categories'])}."
                )
            if bool(plan["backend_rebuilt"]) != expected_rebuild:
                raise AssertionError(
                    f"{entry['step']} expected backend_rebuilt={expected_rebuild}, got {plan['backend_rebuilt']}."
                )
            if int(entry["rgba_bytes"]) != 64 * 64 * 4:
                raise AssertionError(f"{entry['step']} expected 16384 RGBA bytes, got {entry['rgba_bytes']}.")

        output = {
            "benchmark": "incremental_paths",
            "resolution": [64, 64],
            "results": results,
        }
        output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
        print(output_path)
    finally:
        if scene is not None:
            scene.destroy()
        dr.destroy()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
