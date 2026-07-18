"""Compare stage-only and complete ReplicaCAD async camera throughput."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = REPO_ROOT / "ReplicaCAD"
MODULE_DIR = REPO_ROOT / "bin" / "windows-x64"
OUTPUT_DIR = REPO_ROOT / "output" / "replicacad_complete"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default="apt_0")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=96)
    parser.add_argument("--cameras", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--batches", type=int, default=100)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--warmup-batches", type=int, default=20)
    parser.add_argument("--ring-depth", type=int, default=4)
    parser.add_argument(
        "--mode",
        choices=("stage_only", "complete_static", "both"),
        default="complete_static",
    )
    parser.add_argument(
        "--order",
        choices=("stage_first", "complete_first"),
        default="stage_first",
        help="Execution order when --mode both is selected.",
    )
    parser.add_argument("--no-rt-shadows", action="store_true")
    return parser.parse_args()


def query_vram_mb() -> int | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        )
        return int(result.stdout.splitlines()[0].strip())
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def benchmark_batches(scene, camera_ids: list[int], batches: int, ring_depth: int) -> tuple[float, float, int]:
    pending: list[int] = []
    submitted = 0
    busy = 0
    started = time.perf_counter()
    while submitted < batches:
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
    for token in pending:
        scene.read_frame_batch(token)
    elapsed = time.perf_counter() - started
    return len(camera_ids) * batches / elapsed, 1000.0 * elapsed / batches, busy


def prepare_scene(
    scene,
    *,
    scene_path: Path,
    max_cameras: int,
    args: argparse.Namespace,
) -> dict[str, object]:
    baseline_vram = query_vram_mb()
    load_started = time.perf_counter()
    scene.load_scene(str(scene_path))
    load_ms = 1000.0 * (time.perf_counter() - load_started)
    camera = {
        "position": (3.5, 2.0, 3.5),
        "target": (0.0, 1.0, 0.0),
        "up": (0.0, 1.0, 0.0),
        "fov_degrees": 60.0,
        "width": args.width,
        "height": args.height,
        "z_near": 0.1,
        "z_far": 100.0,
    }
    scene.set_camera(**camera)
    while scene.camera_count < max_cameras:
        scene.add_camera(**camera)
    for camera_index in range(scene.camera_count):
        scene.set_camera_at(camera_index, **camera)
    camera_ids = list(range(max_cameras))
    scene.set_ambient((0.03, 0.04, 0.06), (0.01, 0.01, 0.01))
    scene.set_default_light((-0.4, -1.0, -0.6), (1.0, 1.0, 1.0), 2.0)
    rt_shadows = not args.no_rt_shadows
    scene.enable_rt_shadows(rt_shadows)
    scene.enable_shadow_blur(rt_shadows)
    if rt_shadows:
        scene.set_shadow_samples(8)
    scene.set_readback_ring_depth(args.ring_depth)
    scene.render_frame_batch(camera_ids)
    if args.warmup_batches:
        benchmark_batches(scene, camera_ids, args.warmup_batches, args.ring_depth)
    resident_vram = query_vram_mb()
    return {
        "load_ms": load_ms,
        "baseline_vram_mb": baseline_vram,
        "resident_vram_mb": resident_vram,
        "resident_delta_mb": (
            None
            if baseline_vram is None or resident_vram is None
            else max(0, resident_vram - baseline_vram)
        ),
        "native_scene_stats": scene.get_scene_stats(),
    }


def run_trial(
    scene,
    *,
    mode: str,
    num_cameras: int,
    setup: dict[str, object],
    args: argparse.Namespace,
) -> dict[str, object]:
    camera_fps, batch_ms, busy = benchmark_batches(
        scene, list(range(num_cameras)), args.batches, args.ring_depth
    )
    return {
        "mode": mode,
        "num_cameras": num_cameras,
        "camera_fps": camera_fps,
        "batch_ms": batch_ms,
        "ring_backpressure": busy,
        **setup,
    }


def main() -> int:
    args = parse_args()
    if (
        args.width <= 0
        or args.height <= 0
        or args.batches <= 0
        or args.trials <= 0
        or args.warmup_batches < 0
        or any(value <= 0 for value in args.cameras)
    ):
        raise ValueError("Resolution, cameras, batches and trials must be positive.")

    sys.path.insert(0, str(REPO_ROOT / "python"))
    sys.path.insert(0, str(MODULE_DIR))
    from donut_render_py import ReplicaCADManifest, compile_donut_scene
    import DonutRenderPyNative as rr

    manifest = ReplicaCADManifest.from_dataset_root(DATASET_ROOT)
    scene_desc = manifest.parse_scene(args.scene)
    compiled = compile_donut_scene(scene_desc)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    complete_path = OUTPUT_DIR / f"{scene_desc.name}.donut_scene.json"
    compiled.write(complete_path)
    modes = {
        "stage_only": scene_desc.stage.visual_asset.source_path,
        "complete_static": complete_path,
    }
    selected_modes = modes if args.mode == "both" else {args.mode: modes[args.mode]}

    rr.init(
        runtime_dir=str(REPO_ROOT),
        backend="vulkan",
        device_index=-1,
        enable_debug=False,
    )
    try:
        scene = rr.create_scene()
        trial_results = {
            mode: {num_cameras: [] for num_cameras in args.cameras}
            for mode in selected_modes
        }
        ordered_modes = list(selected_modes.items())
        if args.mode == "both" and args.order == "complete_first":
            ordered_modes.reverse()
        for mode, path in ordered_modes:
            setup = prepare_scene(
                scene,
                scene_path=path,
                max_cameras=max(args.cameras),
                args=args,
            )
            for num_cameras in args.cameras:
                if args.warmup_batches:
                    benchmark_batches(
                        scene,
                        list(range(num_cameras)),
                        args.warmup_batches,
                        args.ring_depth,
                    )
                for trial_index in range(args.trials):
                    result = run_trial(
                        scene,
                        mode=mode,
                        num_cameras=num_cameras,
                        setup=setup,
                        args=args,
                    )
                    trial_results[mode][num_cameras].append(result)
                    print(
                        f"[{mode} N={num_cameras} trial "
                        f"{trial_index + 1}/{args.trials}] "
                        f"{result['camera_fps']:.1f} cam-FPS, "
                        f"batch={result['batch_ms']:.3f} ms",
                        flush=True,
                    )
    finally:
        del scene
        rr.destroy()

    aggregates = []
    for mode in selected_modes:
        for num_cameras in args.cameras:
            trials = trial_results[mode][num_cameras]
            camera_fps_values = [float(item["camera_fps"]) for item in trials]
            batch_ms_values = [float(item["batch_ms"]) for item in trials]
            aggregates.append(
                {
                    "mode": mode,
                    "num_cameras": num_cameras,
                    "camera_fps": statistics.median(camera_fps_values),
                    "camera_fps_mean": statistics.mean(camera_fps_values),
                    "camera_fps_std": statistics.pstdev(camera_fps_values),
                    "batch_ms": statistics.median(batch_ms_values),
                    "load_ms": statistics.median(
                        float(item["load_ms"]) for item in trials
                    ),
                    "resident_delta_mb": statistics.median(
                        int(item["resident_delta_mb"])
                        for item in trials
                        if item["resident_delta_mb"] is not None
                    )
                    if any(item["resident_delta_mb"] is not None for item in trials)
                    else None,
                    "native_scene_stats": trials[0]["native_scene_stats"],
                    "trials": trials,
                }
            )

    comparisons = []
    if args.mode == "both":
        for num_cameras in args.cameras:
            stage = next(
                item
                for item in aggregates
                if item["mode"] == "stage_only"
                and item["num_cameras"] == num_cameras
            )
            complete = next(
                item
                for item in aggregates
                if item["mode"] == "complete_static"
                and item["num_cameras"] == num_cameras
            )
            comparisons.append(
                {
                    "num_cameras": num_cameras,
                    "stage_camera_fps": stage["camera_fps"],
                    "complete_camera_fps": complete["camera_fps"],
                    "complete_vs_stage_ratio": complete["camera_fps"]
                    / stage["camera_fps"],
                }
            )

    report = {
        "schema_version": 1,
        "mode": args.mode,
        "scene": scene_desc.name,
        "resolution": [args.width, args.height],
        "rt_shadows": not args.no_rt_shadows,
        "shadow_samples": 0 if args.no_rt_shadows else 8,
        "lighting": {
            "ambient_top": [0.03, 0.04, 0.06],
            "ambient_bottom": [0.01, 0.01, 0.01],
            "direction": [-0.4, -1.0, -0.6],
            "irradiance": 2.0,
        },
        "batches": args.batches,
        "trials": args.trials,
        "order": args.order if args.mode == "both" else None,
        "warmup_batches": args.warmup_batches,
        "ring_depth": args.ring_depth,
        "complete_scene": {
            "unique_models": compiled.model_count,
            "render_instances": compiled.render_instance_count,
            "ordinary_objects": len(scene_desc.objects),
            "articulated_instances": compiled.articulated_instance_count,
            "articulated_links": compiled.articulated_link_count,
            "articulated_visuals": compiled.articulated_visual_count,
            "omitted_articulated_instances": compiled.omitted_articulated_instances,
        },
        "aggregates": aggregates,
        "comparisons": comparisons,
    }
    suffix = "raster" if args.no_rt_shadows else "rt8"
    mode_suffix = f"_{args.order}" if args.mode == "both" else f"_{args.mode}"
    output_path = (
        args.output.resolve()
        if args.output is not None
        else OUTPUT_DIR / f"{scene_desc.name}_parallel_{suffix}{mode_suffix}.json"
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
