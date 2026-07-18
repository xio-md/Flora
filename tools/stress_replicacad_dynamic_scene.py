"""Validate long-running ReplicaCAD articulation updates without reloading assets."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = REPO_ROOT / "ReplicaCAD"
MODULE_DIR = REPO_ROOT / "bin" / "windows-x64"
OUTPUT_DIR = REPO_ROOT / "output" / "replicacad_a3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default="apt_0")
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=48)
    parser.add_argument("--frames", type=int, default=1000)
    parser.add_argument("--warmup-frames", type=int, default=50)
    parser.add_argument("--max-rss-growth-mb", type=float, default=64.0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def process_rss_bytes() -> int | None:
    try:
        import psutil
    except ImportError:
        return None
    return int(psutil.Process().memory_info().rss)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def max_matrix_error(
    lhs: dict[str, list[float]], rhs: dict[str, list[float]]
) -> float:
    return max(
        float(
            np.max(
                np.abs(
                    np.asarray(lhs[name], dtype=np.float64)
                    - np.asarray(rhs[name], dtype=np.float64)
                )
            )
        )
        for name in lhs
    )


def main() -> int:
    args = parse_args()
    if (
        args.width <= 0
        or args.height <= 0
        or args.frames <= 0
        or args.warmup_frames < 0
        or args.max_rss_growth_mb < 0.0
    ):
        raise ValueError("Stress-test dimensions and counts must be non-negative.")

    sys.path.insert(0, str(REPO_ROOT / "python"))
    sys.path.insert(0, str(MODULE_DIR))
    from bench_replicacad_dynamic_parallel import precompute_pose_frames
    from donut_render_py import ReplicaCADManifest, compile_donut_scene
    import DonutRenderPyNative as rr

    manifest = ReplicaCADManifest.from_dataset_root(DATASET_ROOT)
    compiled = compile_donut_scene(manifest.parse_scene(args.scene))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    artifact = compiled.write(OUTPUT_DIR / f"{args.scene}.donut_scene.json")
    pose_names, pose_frames, cpu_fk_reference_ms = precompute_pose_frames(
        compiled, max(args.frames, 256)
    )
    if not pose_names:
        raise RuntimeError("The selected scene has no movable articulation joints.")

    rr.init(
        runtime_dir=str(REPO_ROOT),
        backend="vulkan",
        device_index=-1,
        enable_debug=False,
    )
    scene = None
    try:
        scene = rr.create_scene()
        load_count = 0
        scene.load_scene(str(artifact.scene_path))
        load_count += 1
        scene.set_camera(
            position=(3.5, 2.0, 3.5),
            target=(0.0, 1.0, 0.0),
            up=(0.0, 1.0, 0.0),
            fov_degrees=60.0,
            width=args.width,
            height=args.height,
            z_near=0.05,
            z_far=50.0,
        )
        scene.set_ambient((0.03, 0.04, 0.06), (0.01, 0.01, 0.01))
        scene.set_default_light((-0.4, -1.0, -0.6), (1.0, 1.0, 1.0), 2.0)
        scene.set_shadow_samples(8)

        handles = scene.get_node_handles(pose_names)
        if len(set(handles)) != len(handles):
            raise RuntimeError("Movable control nodes did not resolve to unique handles.")
        initial_handle_count = scene.node_handle_count
        initial_scene_stats = scene.get_scene_stats()
        scene_mtime_ns = artifact.scene_path.stat().st_mtime_ns

        # Compare the new atomic batch API against the compatibility name API.
        comparison_pose = pose_frames[137 % len(pose_frames)]
        scene.enable_rt_shadows(False)
        scene.update_node_transforms_batch(handles, comparison_pose)
        batch_frame = scene.render_frame_batch([0])[0]
        batch_hash = sha256(batch_frame)
        scene.update_node_transforms_batch(handles, pose_frames[23 % len(pose_frames)])
        for name, matrix in zip(pose_names, comparison_pose):
            scene.update_node_transform(name, matrix)
        sequential_frame = scene.render_frame_batch([0])[0]
        sequential_hash = sha256(sequential_frame)
        batch_matches_sequential = batch_frame == sequential_frame

        # Establish a deterministic raster baseline and world-transform reference.
        scene.update_node_transforms_batch(handles, pose_frames[0])
        reference_frame = scene.render_frame_batch([0])[0]
        reference_hash = sha256(reference_frame)
        reference_world = {
            name: scene.get_node_world_transform_by_handle(handle)
            for name, handle in zip(pose_names, handles)
        }

        # Warm all RT resources before taking memory and rebuild baselines.
        scene.enable_rt_shadows(True)
        for frame_index in range(args.warmup_frames):
            scene.update_node_transforms_batch(
                handles, pose_frames[frame_index % len(pose_frames)]
            )
            scene.render_frame_batch([0])
        gc.collect()
        rss_before = process_rss_bytes()

        pose_update_ms = 0.0
        render_batch_ms = 0.0
        refresh_record_ms = 0.0
        shadow_as_record_ms = 0.0
        unexpected_as_builds = 0
        started = time.perf_counter()
        for frame_index in range(args.frames):
            pose_started = time.perf_counter()
            scene.update_node_transforms_batch(
                handles, pose_frames[frame_index % len(pose_frames)]
            )
            pose_update_ms += 1000.0 * (time.perf_counter() - pose_started)

            render_started = time.perf_counter()
            scene.render_frame_batch([0])
            render_batch_ms += 1000.0 * (time.perf_counter() - render_started)
            stats = scene.get_last_frame_stats()
            refresh_record_ms += float(stats["scene_refresh_cpu_ms"])
            shadow_as_record_ms += float(stats["shadow_as_record_cpu_ms"])
            unexpected_as_builds += int(bool(stats["as_built_this_frame"]))
        stress_elapsed_ms = 1000.0 * (time.perf_counter() - started)

        gc.collect()
        rss_after = process_rss_bytes()
        scene.enable_rt_shadows(False)
        scene.update_node_transforms_batch(handles, pose_frames[0])
        restored_frame = scene.render_frame_batch([0])[0]
        restored_hash = sha256(restored_frame)
        restored_world = {
            name: scene.get_node_world_transform_by_handle(handle)
            for name, handle in zip(pose_names, handles)
        }

        transform_restore_error = max_matrix_error(reference_world, restored_world)
        frame_restored = reference_frame == restored_frame
        final_scene_stats = scene.get_scene_stats()
        rss_growth_mb = (
            None
            if rss_before is None or rss_after is None
            else (rss_after - rss_before) / (1024.0 * 1024.0)
        )
        checks = {
            "single_scene_load": load_count == 1,
            "batch_matches_sequential": batch_matches_sequential,
            "restored_raster_frame_matches": frame_restored,
            "restored_world_transform_error_le_1e_6": transform_restore_error <= 1.0e-6,
            "native_handle_count_stable": scene.node_handle_count == initial_handle_count,
            "native_scene_stats_stable": final_scene_stats == initial_scene_stats,
            "scene_artifact_not_rewritten": artifact.scene_path.stat().st_mtime_ns == scene_mtime_ns,
            "no_as_rebuild_after_warmup": unexpected_as_builds == 0,
            "rss_growth_within_limit": (
                rss_growth_mb is None or rss_growth_mb <= args.max_rss_growth_mb
            ),
        }

        remaining_ms = max(
            0.0,
            render_batch_ms - refresh_record_ms - shadow_as_record_ms,
        )
        report = {
            "schema_version": 1,
            "scene": compiled.scene_name,
            "resolution": [args.width, args.height],
            "stress_frames": args.frames,
            "warmup_frames": args.warmup_frames,
            "rt_shadows": True,
            "shadow_samples": 8,
            "movable_joints_per_frame": len(pose_names),
            "cpu_fk_reference_ms_per_pose": cpu_fk_reference_ms,
            "timing_ms_per_frame": {
                "native_pose_update_cpu": pose_update_ms / args.frames,
                "scene_refresh_record_cpu": refresh_record_ms / args.frames,
                "shadow_as_record_cpu": shadow_as_record_ms / args.frames,
                "render_execute_readback_and_python_remaining": remaining_ms / args.frames,
                "render_batch_wall": render_batch_ms / args.frames,
                "full_stress_loop_wall": stress_elapsed_ms / args.frames,
            },
            "consistency": {
                "batch_hash": batch_hash,
                "sequential_hash": sequential_hash,
                "reference_hash": reference_hash,
                "restored_hash": restored_hash,
                "max_restored_world_transform_error": transform_restore_error,
            },
            "resources": {
                "scene_load_count": load_count,
                "node_handle_count_before": initial_handle_count,
                "node_handle_count_after": scene.node_handle_count,
                "unexpected_as_builds_after_warmup": unexpected_as_builds,
                "rss_before_bytes": rss_before,
                "rss_after_bytes": rss_after,
                "rss_growth_mb": rss_growth_mb,
                "max_rss_growth_mb": args.max_rss_growth_mb,
                "native_scene_stats_before": initial_scene_stats,
                "native_scene_stats_after": final_scene_stats,
            },
            "checks": checks,
            "passed": all(checks.values()),
        }
        output_path = (
            args.output.resolve()
            if args.output is not None
            else OUTPUT_DIR / f"{args.scene}_dynamic_stress.json"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        if not report["passed"]:
            failed = [name for name, passed in checks.items() if not passed]
            raise RuntimeError(f"A3 stress checks failed: {failed}; report={output_path}")
        print(
            "PASS: "
            f"frames={args.frames}, joints={len(pose_names)}, "
            f"pose={pose_update_ms / args.frames:.4f} ms, "
            f"refresh={refresh_record_ms / args.frames:.4f} ms, "
            f"AS={shadow_as_record_ms / args.frames:.4f} ms, "
            f"RSS growth={rss_growth_mb} MiB",
            flush=True,
        )
        print(f"Report: {output_path}", flush=True)
        return 0
    finally:
        scene = None
        rr.destroy()


if __name__ == "__main__":
    raise SystemExit(main())
