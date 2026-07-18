"""Benchmark the production single-command-list ReplicaCAD camera batch path."""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENE = REPO_ROOT / "ReplicaCAD" / "stages" / "frl_apartment_stage.glb"
DEFAULT_OUTPUT = REPO_ROOT / "output" / "bench_parallel" / "replicacad_single_cmdlist.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--camera-counts", default="1,2,4,8")
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--global-warmup", type=int, default=20)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--ring-depth", type=int, default=4)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=288)
    parser.add_argument("--rt-shadows", action="store_true")
    return parser.parse_args()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def git_dirty() -> bool:
    try:
        return subprocess.run(
            ["git", "diff", "--quiet"],
            cwd=REPO_ROOT,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode != 0
    except OSError:
        return True


def write_results(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def add_cameras(scene, count: int, width: int, height: int) -> list[int]:
    indices = []
    radius = math.sqrt(3.5**2 + 3.5**2)
    for index in range(count):
        angle = math.radians(45.0) + index * (2.0 * math.pi / count)
        indices.append(
            scene.add_camera(
                position=[radius * math.cos(angle), 2.0, radius * math.sin(angle)],
                target=[0.0, 1.0, 0.0],
                up=[0.0, 1.0, 0.0],
                fov_degrees=60,
                width=width,
                height=height,
                z_near=0.1,
                z_far=100.0,
            )
        )
    return indices


def benchmark_sync(scene, indices: list[int], warmup: int, frames: int) -> dict:
    for _ in range(warmup):
        scene.render_frame_batch(indices)

    start = time.perf_counter()
    for _ in range(frames):
        scene.render_frame_batch(indices)
    elapsed = time.perf_counter() - start
    return {
        "batch_ms": elapsed * 1000.0 / frames,
        "cam_fps": len(indices) * frames / elapsed,
    }


def benchmark_async(scene, indices: list[int], warmup: int, frames: int, ring_depth: int) -> dict:
    for _ in range(warmup):
        token = scene.submit_frame_batch(indices)
        if token == 0:
            raise RuntimeError("Ring unexpectedly busy during async warmup")
        scene.read_frame_batch(token)

    in_flight: list[int] = []
    busy_count = 0
    start = time.perf_counter()
    for _ in range(frames):
        if len(in_flight) == ring_depth:
            scene.read_frame_batch(in_flight.pop(0))

        token = scene.submit_frame_batch(indices)
        if token == 0:
            busy_count += 1
            if not in_flight:
                raise RuntimeError("Ring reported busy with no tracked in-flight batch")
            scene.read_frame_batch(in_flight.pop(0))
            token = scene.submit_frame_batch(indices)
        if token == 0:
            raise RuntimeError("Ring remained busy after releasing the oldest batch")
        in_flight.append(token)

    for token in in_flight:
        scene.read_frame_batch(token)
    elapsed = time.perf_counter() - start
    return {
        "batch_ms": elapsed * 1000.0 / frames,
        "cam_fps": len(indices) * frames / elapsed,
        "busy_count": busy_count,
    }


def summarize(samples: list[dict]) -> dict:
    summary = {
        "batch_ms": statistics.median(sample["batch_ms"] for sample in samples),
        "cam_fps": statistics.median(sample["cam_fps"] for sample in samples),
        "samples": samples,
    }
    if "busy_count" in samples[0]:
        summary["busy_count"] = sum(sample["busy_count"] for sample in samples)
    return summary


def main() -> int:
    args = parse_args()
    camera_counts = sorted({int(value) for value in args.camera_counts.split(",")})
    if not camera_counts or camera_counts[0] < 1:
        raise ValueError("--camera-counts must contain positive integers")
    if args.frames < 1 or args.warmup < 0 or args.global_warmup < 0 or args.trials < 1:
        raise ValueError("--frames/--trials must be positive and warmup values must be non-negative")
    if args.ring_depth < 2:
        raise ValueError("--ring-depth must be at least 2")
    if not args.scene.is_file():
        raise FileNotFoundError(f"ReplicaCAD scene not found: {args.scene}")

    module_dir = REPO_ROOT / "bin" / "windows-x64"
    sys.path.insert(0, str(module_dir))
    import DonutRenderPyNative as rr

    payload = {
        "metadata": {
            "commit": git_commit(),
            "working_tree_dirty": git_dirty(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "scene": portable_path(args.scene),
            "scene_kind": "stage_glb",
            "resolution": [args.width, args.height],
            "frames": args.frames,
            "warmup": args.warmup,
            "global_warmup": args.global_warmup,
            "trials": args.trials,
            "ring_depth": args.ring_depth,
            "rt_shadows": args.rt_shadows,
            "submission": "single_cmdlist",
            "endpoint": "cpu_rgba8_e2e",
        },
        "results": [],
    }
    write_results(args.output, payload)

    rr.init(runtime_dir=str(REPO_ROOT), backend="vulkan", device_index=-1, enable_debug=False)
    try:
        scene = rr.create_scene()
        scene.load_scene(str(args.scene))
        scene.set_default_light(direction=[-0.4, -1.0, -0.6], color=[1, 1, 1], irradiance=2.0)
        scene.set_ambient(top_rgb=[0.03, 0.04, 0.06], bottom_rgb=[0.01, 0.01, 0.01])
        scene.enable_rt_shadows(args.rt_shadows)
        scene.enable_shadow_blur(args.rt_shadows)
        scene.set_shadow_samples(8)
        scene.set_readback_ring_depth(args.ring_depth)

        all_cameras = add_cameras(scene, max(camera_counts), args.width, args.height)
        if args.global_warmup:
            print(
                f"[warmup] C={len(all_cameras)}, batches={args.global_warmup}",
                flush=True,
            )
            for _ in range(args.global_warmup):
                scene.render_frame_batch(all_cameras)

        for case_index, camera_count in enumerate(camera_counts, start=1):
            indices = all_cameras[:camera_count]
            print(
                f"[{case_index}/{len(camera_counts)}] C={camera_count}, "
                f"{args.width}x{args.height}, RT={args.rt_shadows}",
                flush=True,
            )
            sync_samples: list[dict] = []
            async_samples: list[dict] = []
            for trial in range(args.trials):
                print(f"  trial {trial + 1}/{args.trials}", flush=True)
                if trial % 2 == 0:
                    sync_samples.append(benchmark_sync(scene, indices, args.warmup, args.frames))
                    async_samples.append(
                        benchmark_async(scene, indices, args.warmup, args.frames, args.ring_depth)
                    )
                else:
                    async_samples.append(
                        benchmark_async(scene, indices, args.warmup, args.frames, args.ring_depth)
                    )
                    sync_samples.append(benchmark_sync(scene, indices, args.warmup, args.frames))

            sync = summarize(sync_samples)
            async_result = summarize(async_samples)
            case = {
                "camera_count": camera_count,
                "sync": sync,
                "async_e2e": async_result,
            }
            payload["results"].append(case)
            write_results(args.output, payload)
            print(
                f"  sync={sync['cam_fps']:.1f} cam-FPS, "
                f"async={async_result['cam_fps']:.1f} cam-FPS, "
                f"busy={async_result['busy_count']}",
                flush=True,
            )

        reference = scene.render_frame(all_cameras[0])
        rgb = np.frombuffer(reference, np.uint8).reshape(args.height, args.width, 4)[:, :, :3]
        reference_path = args.output.with_name(args.output.stem + "_reference.png")
        Image.fromarray(rgb, "RGB").save(reference_path)
        payload["reference_image"] = portable_path(reference_path)
        write_results(args.output, payload)
        print(f"Results: {args.output}", flush=True)
        print(f"Reference: {reference_path}", flush=True)
        return 0
    finally:
        rr.destroy()


if __name__ == "__main__":
    raise SystemExit(main())
