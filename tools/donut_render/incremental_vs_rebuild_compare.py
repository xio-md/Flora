from __future__ import annotations

import argparse
import json
import sys
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


def _create_scene(dr, *, width: int, height: int, env_rgb: tuple[float, float, float], camera_pose: np.ndarray, transform: np.ndarray):
    scene = dr.create_scene()
    scene.init(
        dr.Render(
            name="incremental-compare",
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
        pose=dr.MatrixTransform(np.asarray(camera_pose, dtype=np.float32)),
        film=dr.Film((width, height)),
        filter=dr.Filter(1.0),
        spp=1,
        fov=45.0,
    )
    shape = dr.RigidShape(
        name="quad",
        vertices=_quad_vertices(1.0),
        triangles=_quad_triangles(),
        transform=dr.MatrixTransform(np.asarray(transform, dtype=np.float32)),
        surface="mat",
    )

    scene.update_environment(
        dr.Environment(
            name="env",
            emission=dr.ColorTexture(env_rgb),
        )
    )
    scene.update_surface(surface)
    scene.update_shape(shape)
    scene.update_camera(camera, denoise=False)
    return scene, camera, shape


def _render_report(scene, camera, *, time_value: float) -> tuple[dict[str, object], bytes]:
    preview = _serialize_plan(scene.preview_update_plan(time=time_value))
    scene.update_scene(time=time_value)
    report = scene.get_update_stats()
    rgba = scene.render_frame(camera)
    return {
        "preview": preview,
        "report_mode": str(report["mode"]),
        "report_plan": _serialize_plan(report["plan"]),
        "duration_ms": float(report["duration_ms"]),
        "rgba_bytes": len(rgba),
    }, rgba


def _compare_rgba(lhs: bytes, rhs: bytes) -> dict[str, object]:
    lhs_array = np.frombuffer(lhs, dtype=np.uint8)
    rhs_array = np.frombuffer(rhs, dtype=np.uint8)
    if lhs_array.shape != rhs_array.shape:
        raise AssertionError(f"RGBA shape mismatch: {lhs_array.shape} vs {rhs_array.shape}.")
    diff = np.abs(lhs_array.astype(np.int16) - rhs_array.astype(np.int16))
    return {
        "equal": bool(np.array_equal(lhs_array, rhs_array)),
        "max_abs_diff": int(diff.max(initial=0)),
        "mean_abs_diff": float(diff.mean()) if diff.size else 0.0,
        "changed_bytes": int(np.count_nonzero(diff)),
    }


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "python"))

    import DonutRenderPy as dr

    parser = argparse.ArgumentParser(description="Compare incremental renders against equivalent full rebuild renders.")
    parser.add_argument("--module-dir", type=Path, default=repo_root / "bin" / "windows-x64")
    parser.add_argument("--runtime-dir", type=Path, default=repo_root)
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / ".temp" / "incremental_vs_rebuild_compare.json",
    )
    args = parser.parse_args()

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    width = 64
    height = 64
    initial_env = (0.08, 0.10, 0.14)
    initial_pose = _camera_pose(3.0)
    identity = np.eye(4, dtype=np.float32)

    scenarios = [
        {
            "name": "camera_only",
            "time": 1.0,
            "expected_incremental_mode": "incremental_camera_environment",
            "env_rgb": initial_env,
            "camera_pose": _camera_pose(3.2, x_offset=0.15),
            "transform": identity,
        },
        {
            "name": "environment_only",
            "time": 2.0,
            "expected_incremental_mode": "incremental_camera_environment",
            "env_rgb": (0.15, 0.18, 0.22),
            "camera_pose": initial_pose,
            "transform": identity,
        },
        {
            "name": "transform_only",
            "time": 3.0,
            "expected_incremental_mode": "incremental_camera_transform_environment",
            "env_rgb": initial_env,
            "camera_pose": initial_pose,
            "transform": np.array(
                (
                    (1.0, 0.0, 0.0, 0.25),
                    (0.0, 1.0, 0.0, 0.0),
                    (0.0, 0.0, 1.0, 0.0),
                    (0.0, 0.0, 0.0, 1.0),
                ),
                dtype=np.float32,
            ),
        },
    ]

    dr.init(
        context_path=repo_root,
        runtime_dir=args.runtime_dir,
        module_dir=args.module_dir,
        backend="vulkan",
        device_index=-1,
        log_level=dr.LogLevel.WARNING,
    )

    results: list[dict[str, object]] = []
    try:
        for scenario in scenarios:
            scene = None
            reference_scene = None
            try:
                scene, camera, shape = _create_scene(
                    dr,
                    width=width,
                    height=height,
                    env_rgb=initial_env,
                    camera_pose=initial_pose,
                    transform=identity,
                )
                initial_report, _initial_rgba = _render_report(scene, camera, time_value=0.0)
                if initial_report["report_mode"] != "full_rebuild":
                    raise AssertionError(f"{scenario['name']} expected initial build to be full_rebuild.")

                if scenario["name"] == "camera_only":
                    camera.update(pose=dr.MatrixTransform(np.asarray(scenario["camera_pose"], dtype=np.float32)))
                    scene.update_camera(camera, denoise=False)
                elif scenario["name"] == "environment_only":
                    scene.update_environment(
                        dr.Environment(
                            name="env",
                            emission=dr.ColorTexture(tuple(scenario["env_rgb"])),
                        )
                    )
                elif scenario["name"] == "transform_only":
                    shape.update(transform=dr.MatrixTransform(np.asarray(scenario["transform"], dtype=np.float32)))
                    scene.update_shape(shape)
                else:
                    raise AssertionError(f"Unknown scenario {scenario['name']}.")

                incremental_report, incremental_rgba = _render_report(scene, camera, time_value=float(scenario["time"]))
                if incremental_report["report_mode"] != scenario["expected_incremental_mode"]:
                    raise AssertionError(
                        f"{scenario['name']} expected incremental mode {scenario['expected_incremental_mode']}, got {incremental_report['report_mode']}."
                    )

                reference_scene, reference_camera, _reference_shape = _create_scene(
                    dr,
                    width=width,
                    height=height,
                    env_rgb=tuple(scenario["env_rgb"]),
                    camera_pose=np.asarray(scenario["camera_pose"], dtype=np.float32),
                    transform=np.asarray(scenario["transform"], dtype=np.float32),
                )
                rebuild_report, rebuild_rgba = _render_report(reference_scene, reference_camera, time_value=float(scenario["time"]))
                if rebuild_report["report_mode"] != "full_rebuild":
                    raise AssertionError(f"{scenario['name']} expected rebuild reference to be full_rebuild.")

                diff_metrics = _compare_rgba(incremental_rgba, rebuild_rgba)
                if not diff_metrics["equal"]:
                    raise AssertionError(
                        f"{scenario['name']} incremental render diverged from rebuild reference: "
                        f"max_abs_diff={diff_metrics['max_abs_diff']}, changed_bytes={diff_metrics['changed_bytes']}."
                    )
                if int(incremental_report["rgba_bytes"]) != width * height * 4:
                    raise AssertionError(f"{scenario['name']} expected {width * height * 4} bytes, got {incremental_report['rgba_bytes']}.")

                results.append(
                    {
                        "scenario": str(scenario["name"]),
                        "incremental": incremental_report,
                        "reference_rebuild": rebuild_report,
                        "comparison": diff_metrics,
                    }
                )
            finally:
                if scene is not None:
                    scene.destroy()
                if reference_scene is not None:
                    reference_scene.destroy()
    finally:
        dr.destroy()

    output = {
        "comparison": "incremental_vs_rebuild",
        "resolution": [width, height],
        "results": results,
    }
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
