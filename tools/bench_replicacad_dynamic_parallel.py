"""Benchmark static vs articulated-pose ReplicaCAD async multi-camera throughput."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = REPO_ROOT / "ReplicaCAD"
MODULE_DIR = REPO_ROOT / "bin" / "windows-x64"
OUTPUT_DIR = REPO_ROOT / "output" / "replicacad_a3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default="apt_0")
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=96)
    parser.add_argument("--cameras", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--batches", type=int, default=200)
    parser.add_argument("--warmup-batches", type=int, default=50)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--ring-depth", type=int, default=4)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def flatten_matrix(matrix: tuple[tuple[float, ...], ...]) -> list[float]:
    return [float(value) for row in matrix for value in row]


def animated_joint_positions(compiled, frame_index: int) -> dict[int, dict[str, float]]:
    positions: dict[int, dict[str, float]] = {}
    joint_index = 0
    phase = frame_index * 0.17
    for articulation in compiled.articulations:
        articulation_positions: dict[str, float] = {}
        for link in articulation.links:
            joint = link.joint
            if joint is None or joint.joint_type.lower() == "fixed":
                continue
            lower = -0.5 if joint.limit_lower is None else joint.limit_lower
            upper = 0.5 if joint.limit_upper is None else joint.limit_upper
            midpoint = 0.5 * (lower + upper)
            amplitude = 0.45 * (upper - lower)
            articulation_positions[joint.name] = midpoint + amplitude * math.sin(
                phase + joint_index * 0.31
            )
            joint_index += 1
        if articulation_positions:
            positions[articulation.instance_id] = articulation_positions
    return positions


def precompute_pose_frames(compiled, frame_count: int) -> tuple[
    tuple[str, ...], list[list[list[float]]], float
]:
    names: tuple[str, ...] | None = None
    frames: list[list[list[float]]] = []
    started = time.perf_counter()
    for frame_index in range(frame_count):
        frame_names, matrices = compiled.joint_transform_updates(
            animated_joint_positions(compiled, frame_index)
        )
        if names is None:
            names = frame_names
        elif frame_names != names:
            raise RuntimeError("Dynamic joint update topology changed between frames.")
        frames.append([flatten_matrix(matrix) for matrix in matrices])
    elapsed_ms = 1000.0 * (time.perf_counter() - started)
    return names or (), frames, elapsed_ms / frame_count


def benchmark_batches(
    scene,
    compiled,
    camera_ids: list[int],
    batches: int,
    ring_depth: int,
    *,
    dynamic: bool,
    update_handles: list[int],
    pose_frames: list[list[list[float]]],
    phase_offset: int,
) -> dict[str, float | int]:
    pending: list[int] = []
    submitted = 0
    busy = 0
    update_ms = 0.0
    started = time.perf_counter()
    while submitted < batches:
        if dynamic:
            matrix_values = pose_frames[(phase_offset + submitted) % len(pose_frames)]
            update_start = time.perf_counter()
            scene.update_node_transforms_batch(
                update_handles,
                matrix_values,
            )
            update_ms += 1000.0 * (time.perf_counter() - update_start)
        token = scene.submit_frame_batch(camera_ids)
        if token:
            pending.append(token)
            submitted += 1
            if len(pending) >= ring_depth:
                scene.read_frame_batch(pending.pop(0))
        else:
            busy += 1
            if pending:
                scene.read_frame_batch(pending.pop(0))
            else:
                raise RuntimeError("Renderer reported ring backpressure with no pending batch.")
    for token in pending:
        scene.read_frame_batch(token)
    elapsed = time.perf_counter() - started
    return {
        "camera_fps": len(camera_ids) * batches / elapsed,
        "batch_ms": 1000.0 * elapsed / batches,
        "native_pose_update_ms_per_batch": update_ms / batches,
        "ring_backpressure": busy,
    }


def main() -> int:
    args = parse_args()
    if (
        args.width <= 0
        or args.height <= 0
        or args.batches <= 0
        or args.warmup_batches < 0
        or args.trials <= 0
        or args.ring_depth < 2
        or not args.cameras
        or min(args.cameras) <= 0
    ):
        raise ValueError("Benchmark dimensions, counts and ring depth are invalid.")

    sys.path.insert(0, str(REPO_ROOT / "python"))
    sys.path.insert(0, str(MODULE_DIR))
    from donut_render_py import ReplicaCADManifest, compile_donut_scene
    import DonutRenderPyNative as rr

    manifest = ReplicaCADManifest.from_dataset_root(DATASET_ROOT)
    compiled = compile_donut_scene(manifest.parse_scene(args.scene))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    artifact = compiled.write(OUTPUT_DIR / f"{args.scene}.donut_scene.json")

    initial_names, pose_frames, cpu_fk_reference_ms = precompute_pose_frames(
        compiled, 512
    )
    movable_joint_count = len(initial_names)

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
        scene.enable_rt_shadows(True)
        scene.enable_shadow_blur(True)
        scene.set_shadow_samples(8)
        scene.set_readback_ring_depth(args.ring_depth)
        update_handles = scene.get_node_handles(initial_names)

        scene.render_frame_batch(list(range(max(args.cameras))))
        results: list[dict[str, object]] = []
        phase_offset = 0
        for num_cameras in args.cameras:
            camera_ids = list(range(num_cameras))
            if args.warmup_batches:
                benchmark_batches(
                    scene,
                    compiled,
                    camera_ids,
                    args.warmup_batches,
                    args.ring_depth,
                    dynamic=True,
                    update_handles=update_handles,
                    pose_frames=pose_frames,
                    phase_offset=phase_offset,
                )
                phase_offset += args.warmup_batches
            for trial in range(args.trials):
                mode_order = (False, True) if trial % 2 == 0 else (True, False)
                for dynamic in mode_order:
                    result = benchmark_batches(
                        scene,
                        compiled,
                        camera_ids,
                        args.batches,
                        args.ring_depth,
                        dynamic=dynamic,
                        update_handles=update_handles,
                        pose_frames=pose_frames,
                        phase_offset=phase_offset,
                    )
                    if dynamic:
                        phase_offset += args.batches
                    result.update(
                        {
                            "mode": "dynamic_pose" if dynamic else "static_pose",
                            "num_cameras": num_cameras,
                            "trial": trial + 1,
                        }
                    )
                    results.append(result)
                    print(
                        f"[N={num_cameras} trial={trial + 1} {result['mode']}] "
                        f"{result['camera_fps']:.1f} cam-FPS, "
                        f"batch={result['batch_ms']:.3f} ms, "
                        f"pose={result['native_pose_update_ms_per_batch']:.3f} ms",
                        flush=True,
                    )

        native_stats = scene.get_scene_stats()
        native_node_handles = scene.node_handle_count
    finally:
        scene = None
        rr.destroy()

    aggregates: list[dict[str, object]] = []
    for num_cameras in args.cameras:
        modes: dict[str, dict[str, object]] = {}
        for mode in ("static_pose", "dynamic_pose"):
            samples = [
                result
                for result in results
                if result["num_cameras"] == num_cameras and result["mode"] == mode
            ]
            modes[mode] = {
                "camera_fps": statistics.median(
                    float(sample["camera_fps"]) for sample in samples
                ),
                "batch_ms": statistics.median(
                    float(sample["batch_ms"]) for sample in samples
                ),
                "native_pose_update_ms_per_batch": statistics.median(
                    float(sample["native_pose_update_ms_per_batch"])
                    for sample in samples
                ),
            }
        dynamic_batch_ms = float(modes["dynamic_pose"]["batch_ms"])
        aggregates.append(
            {
                "num_cameras": num_cameras,
                "static_pose": modes["static_pose"],
                "dynamic_pose": modes["dynamic_pose"],
                "dynamic_vs_static_fps_ratio": (
                    float(modes["dynamic_pose"]["camera_fps"])
                    / float(modes["static_pose"]["camera_fps"])
                ),
                "dynamic_with_cpu_fk_estimate": {
                    "batch_ms": dynamic_batch_ms + cpu_fk_reference_ms,
                    "camera_fps": (
                        num_cameras * 1000.0
                        / (dynamic_batch_ms + cpu_fk_reference_ms)
                    ),
                },
            }
        )

    report = {
        "schema_version": 1,
        "scene": compiled.scene_name,
        "resolution": [args.width, args.height],
        "rt_shadows": True,
        "shadow_samples": 8,
        "batches": args.batches,
        "warmup_batches": args.warmup_batches,
        "trials": args.trials,
        "ring_depth": args.ring_depth,
        "movable_joints_per_batch": movable_joint_count,
        "cpu_fk_reference_ms_per_pose": cpu_fk_reference_ms,
        "scene_summary": {
            "unique_models": compiled.model_count,
            "render_instances": compiled.render_instance_count,
            "articulated_instances": compiled.articulated_instance_count,
            "articulated_links": compiled.articulated_link_count,
            "articulated_visuals": compiled.articulated_visual_count,
            "native_node_handles": native_node_handles,
            "native_scene_stats": native_stats,
        },
        "aggregates": aggregates,
        "trials_data": results,
    }
    output_path = (
        args.output.resolve()
        if args.output is not None
        else OUTPUT_DIR / f"{args.scene}_dynamic_parallel_rt8.json"
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
