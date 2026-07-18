"""Benchmark ReplicaCAD Color-only and five-product sensor throughput."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = REPO_ROOT / "ReplicaCAD"
MODULE_DIR = REPO_ROOT / "bin" / "windows-x64"
OUTPUT_DIR = REPO_ROOT / "output" / "replicacad_a4"
ALL_PRODUCTS = ("color", "depth", "normal", "instance", "semantic")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default="apt_0")
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=96)
    parser.add_argument("--cameras", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--batches", type=int, default=200)
    parser.add_argument("--warmup-batches", type=int, default=30)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--rt-shadows", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def payload_bytes(frames: list[dict[str, object]]) -> int:
    return sum(
        len(value)
        for frame in frames
        for name, value in frame.items()
        if name not in ("width", "height")
    )


def run_batches(
    scene,
    camera_ids: list[int],
    products: tuple[str, ...],
    batches: int,
    update_handles: list[int],
    pose_frames: list[list[list[float]]],
    phase_offset: int,
) -> dict[str, float | int]:
    refresh_record_ms = 0.0
    sensor_record_ms = 0.0
    pose_write_ms = 0.0
    total_payload_bytes = 0
    started = time.perf_counter()
    for batch_index in range(batches):
        pose_started = time.perf_counter()
        scene.update_node_transforms_batch(
            update_handles,
            pose_frames[(phase_offset + batch_index) % len(pose_frames)],
        )
        pose_write_ms += 1000.0 * (time.perf_counter() - pose_started)
        frames = scene.render_sensor_batch(camera_ids, list(products))
        total_payload_bytes += payload_bytes(frames)
        stats = scene.get_last_frame_stats()
        refresh_record_ms += float(stats["scene_refresh_cpu_ms"])
        sensor_record_ms += float(stats["sensor_record_cpu_ms"])
    elapsed_ms = 1000.0 * (time.perf_counter() - started)
    camera_frames = batches * len(camera_ids)
    return {
        "batches": batches,
        "camera_frames": camera_frames,
        "elapsed_ms": elapsed_ms,
        "batch_ms": elapsed_ms / batches,
        "camera_fps": camera_frames * 1000.0 / elapsed_ms,
        "pose_write_ms_per_batch": pose_write_ms / batches,
        "scene_refresh_record_ms_per_batch": refresh_record_ms / batches,
        "sensor_record_ms_per_batch": sensor_record_ms / batches,
        "payload_bytes_per_camera": total_payload_bytes // camera_frames,
    }


def main() -> int:
    args = parse_args()
    if (
        args.width <= 0
        or args.height <= 0
        or args.batches <= 0
        or args.warmup_batches < 0
        or args.trials <= 0
        or not args.cameras
        or min(args.cameras) <= 0
    ):
        raise ValueError("Benchmark dimensions and counts must be positive.")

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
        compiled, 512
    )

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
        camera = {
            "position": (3.5, 2.0, 3.5),
            "target": (0.0, 1.0, 0.0),
            "up": (0.0, 1.0, 0.0),
            "fov_degrees": 60.0,
            "width": args.width,
            "height": args.height,
            "z_near": 0.05,
            "z_far": 50.0,
        }
        scene.set_camera(**camera)
        while scene.camera_count < max(args.cameras):
            scene.add_camera(**camera)
        scene.set_ambient((0.03, 0.04, 0.06), (0.01, 0.01, 0.01))
        scene.set_default_light((-0.4, -1.0, -0.6), (1.0, 1.0, 1.0), 2.0)
        scene.enable_rt_shadows(args.rt_shadows)
        scene.enable_shadow_blur(True)
        scene.set_shadow_samples(8)
        update_handles = scene.get_node_handles(pose_names)

        results: list[dict[str, object]] = []
        phase_offset = 0
        product_modes = {
            "color_only": ("color",),
            "all_products": ALL_PRODUCTS,
        }
        for camera_count in args.cameras:
            camera_ids = list(range(camera_count))
            if args.warmup_batches:
                run_batches(
                    scene,
                    camera_ids,
                    ALL_PRODUCTS,
                    args.warmup_batches,
                    update_handles,
                    pose_frames,
                    phase_offset,
                )
                phase_offset += args.warmup_batches
            for trial in range(args.trials):
                mode_names = list(product_modes)
                if trial % 2:
                    mode_names.reverse()
                for mode_name in mode_names:
                    result = run_batches(
                        scene,
                        camera_ids,
                        product_modes[mode_name],
                        args.batches,
                        update_handles,
                        pose_frames,
                        phase_offset,
                    )
                    phase_offset += args.batches
                    result.update(
                        {
                            "mode": mode_name,
                            "products": list(product_modes[mode_name]),
                            "num_cameras": camera_count,
                            "trial": trial + 1,
                        }
                    )
                    results.append(result)
                    print(
                        f"[N={camera_count} trial={trial + 1} {mode_name}] "
                        f"{result['camera_fps']:.1f} cam-FPS, "
                        f"batch={result['batch_ms']:.3f} ms",
                        flush=True,
                    )
        scene_stats = scene.get_scene_stats()
    finally:
        scene = None
        rr.destroy()

    aggregates: list[dict[str, object]] = []
    for camera_count in args.cameras:
        modes = {}
        for mode_name in ("color_only", "all_products"):
            samples = [
                result
                for result in results
                if result["num_cameras"] == camera_count
                and result["mode"] == mode_name
            ]
            modes[mode_name] = {
                key: statistics.median(float(sample[key]) for sample in samples)
                for key in (
                    "camera_fps",
                    "batch_ms",
                    "pose_write_ms_per_batch",
                    "scene_refresh_record_ms_per_batch",
                    "sensor_record_ms_per_batch",
                    "payload_bytes_per_camera",
                )
            }
        aggregates.append(
            {
                "num_cameras": camera_count,
                **modes,
                "all_vs_color_fps_ratio": (
                    modes["all_products"]["camera_fps"]
                    / modes["color_only"]["camera_fps"]
                ),
                "all_products_extra_batch_ms": (
                    modes["all_products"]["batch_ms"]
                    - modes["color_only"]["batch_ms"]
                ),
            }
        )

    report = {
        "schema_version": 1,
        "scene": compiled.scene_name,
        "resolution": [args.width, args.height],
        "rt_shadows": args.rt_shadows,
        "shadow_samples": 8,
        "dynamic_pose": True,
        "movable_joints_per_batch": len(pose_names),
        "cpu_fk_reference_ms_per_pose": cpu_fk_reference_ms,
        "batches": args.batches,
        "warmup_batches": args.warmup_batches,
        "trials": args.trials,
        "scene_stats": scene_stats,
        "aggregates": aggregates,
        "trials_data": results,
    }
    output_path = (
        args.output.resolve()
        if args.output is not None
        else OUTPUT_DIR / f"{args.scene}_multimodal_benchmark.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"PASS: report={output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
