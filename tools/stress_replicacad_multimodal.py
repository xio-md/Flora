"""Stress aligned ReplicaCAD multimodal rendering with dynamic joint poses."""

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
OUTPUT_DIR = REPO_ROOT / "output" / "replicacad_a4"
PRODUCTS = ("color", "depth", "normal", "instance", "semantic")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default="apt_0")
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=48)
    parser.add_argument("--frames", type=int, default=1000)
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument("--max-rss-growth-mb", type=float, default=64.0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def process_rss_bytes() -> int | None:
    try:
        import psutil
    except ImportError:
        return None
    return int(psutil.Process().memory_info().rss)


def frame_hashes(frame) -> dict[str, str]:
    return {
        product: hashlib.sha256(getattr(frame, product).tobytes()).hexdigest()
        for product in PRODUCTS
    }


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
    from rtxns_genesis_style.sensor import decode_sensor_frame
    import DonutRenderPyNative as rr

    manifest = ReplicaCADManifest.from_dataset_root(DATASET_ROOT)
    compiled = compile_donut_scene(manifest.parse_scene(args.scene))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    artifact = compiled.write(OUTPUT_DIR / f"{args.scene}.donut_scene.json")
    pose_names, pose_frames, cpu_fk_reference_ms = precompute_pose_frames(
        compiled, max(args.frames, 256)
    )
    semantic_lut = np.zeros(len(compiled.sensor_labels) + 1, dtype=np.uint32)
    for label in compiled.sensor_labels:
        semantic_lut[label.instance_id] = label.semantic_id

    rr.init(
        runtime_dir=str(REPO_ROOT),
        backend="vulkan",
        device_index=-1,
        enable_debug=False,
    )
    scene = None
    try:
        scene = rr.create_scene()
        scene.load_scene(str(artifact.scene_path))
        compiled.configure_sensor_labels(scene)
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
        scene.enable_rt_shadows(False)
        update_handles = scene.get_node_handles(pose_names)
        initial_handle_count = scene.node_handle_count
        initial_scene_stats = scene.get_scene_stats()
        scene_mtime_ns = artifact.scene_path.stat().st_mtime_ns

        for frame_index in range(args.warmup_frames):
            scene.update_node_transforms_batch(
                update_handles, pose_frames[frame_index % len(pose_frames)]
            )
            scene.render_sensor_batch([0], list(PRODUCTS))

        scene.update_node_transforms_batch(update_handles, pose_frames[0])
        reference = decode_sensor_frame(
            scene.render_sensor_batch([0], list(PRODUCTS))[0]
        )
        reference_hashes = frame_hashes(reference)
        gc.collect()
        rss_before = process_rss_bytes()

        pose_write_ms = 0.0
        render_ms = 0.0
        refresh_record_ms = 0.0
        sensor_record_ms = 0.0
        mask_mismatch_frames = 0
        semantic_mismatch_frames = 0
        unknown_instance_frames = 0
        started = time.perf_counter()
        for frame_index in range(args.frames):
            pose_started = time.perf_counter()
            scene.update_node_transforms_batch(
                update_handles, pose_frames[frame_index % len(pose_frames)]
            )
            pose_write_ms += 1000.0 * (time.perf_counter() - pose_started)

            render_started = time.perf_counter()
            frame = decode_sensor_frame(
                scene.render_sensor_batch([0], list(PRODUCTS))[0]
            )
            render_ms += 1000.0 * (time.perf_counter() - render_started)
            depth_valid = frame.depth > 0.0
            normal_valid = np.linalg.norm(frame.normal, axis=2) > 0.5
            instance_valid = frame.instance > 0
            if not (
                np.array_equal(depth_valid, normal_valid)
                and np.array_equal(depth_valid, instance_valid)
            ):
                mask_mismatch_frames += 1
            if int(frame.instance.max()) >= semantic_lut.size:
                unknown_instance_frames += 1
            elif not np.array_equal(frame.semantic, semantic_lut[frame.instance]):
                semantic_mismatch_frames += 1
            stats = scene.get_last_frame_stats()
            refresh_record_ms += float(stats["scene_refresh_cpu_ms"])
            sensor_record_ms += float(stats["sensor_record_cpu_ms"])
        stress_elapsed_ms = 1000.0 * (time.perf_counter() - started)

        gc.collect()
        rss_after = process_rss_bytes()
        scene.update_node_transforms_batch(update_handles, pose_frames[0])
        restored = decode_sensor_frame(
            scene.render_sensor_batch([0], list(PRODUCTS))[0]
        )
        restored_hashes = frame_hashes(restored)
        final_scene_stats = scene.get_scene_stats()
        rss_growth_mb = (
            None
            if rss_before is None or rss_after is None
            else (rss_after - rss_before) / (1024.0 * 1024.0)
        )
        checks = {
            "all_valid_masks_aligned": mask_mismatch_frames == 0,
            "all_instance_ids_registered": unknown_instance_frames == 0,
            "all_semantic_mappings_exact": semantic_mismatch_frames == 0,
            "all_products_restore_exactly": reference_hashes == restored_hashes,
            "native_handle_count_stable": scene.node_handle_count == initial_handle_count,
            "native_scene_stats_stable": final_scene_stats == initial_scene_stats,
            "scene_artifact_not_rewritten": artifact.scene_path.stat().st_mtime_ns
            == scene_mtime_ns,
            "rss_growth_within_limit": (
                rss_growth_mb is None or rss_growth_mb <= args.max_rss_growth_mb
            ),
        }
        report = {
            "schema_version": 1,
            "scene": compiled.scene_name,
            "resolution": [args.width, args.height],
            "products": list(PRODUCTS),
            "stress_frames": args.frames,
            "warmup_frames": args.warmup_frames,
            "rt_shadows": False,
            "movable_joints_per_frame": len(pose_names),
            "sensor_label_count": len(compiled.sensor_labels),
            "cpu_fk_reference_ms_per_pose": cpu_fk_reference_ms,
            "timing_ms_per_frame": {
                "native_pose_write_cpu": pose_write_ms / args.frames,
                "scene_refresh_record_cpu": refresh_record_ms / args.frames,
                "sensor_record_cpu": sensor_record_ms / args.frames,
                "render_execute_readback_decode_wall": render_ms / args.frames,
                "full_stress_loop_wall": stress_elapsed_ms / args.frames,
            },
            "consistency": {
                "mask_mismatch_frames": mask_mismatch_frames,
                "semantic_mismatch_frames": semantic_mismatch_frames,
                "unknown_instance_frames": unknown_instance_frames,
                "reference_hashes": reference_hashes,
                "restored_hashes": restored_hashes,
            },
            "resources": {
                "node_handle_count_before": initial_handle_count,
                "node_handle_count_after": scene.node_handle_count,
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
            else OUTPUT_DIR / f"{args.scene}_multimodal_stress.json"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        if not report["passed"]:
            failed = [name for name, passed in checks.items() if not passed]
            raise RuntimeError(f"A4 stress checks failed: {failed}; report={output_path}")
        print(
            "PASS: "
            f"frames={args.frames}, labels={len(compiled.sensor_labels)}, "
            f"render={render_ms / args.frames:.4f} ms, "
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
