"""Parallel rendering POC — multi-process throughput benchmark.

Spawns N worker processes, each loading an independent RTXNS Vulkan device and
rendering Bistro from a different camera angle. Measures wall-clock throughput
(FPS) to evaluate the value of process-level parallel rendering.

This is "Plan A" from the parallel-render feasibility analysis: no C++ changes,
pure Python multiprocessing wrapper. If N=4 yields >2x throughput vs N=1, it
justifies investing in "Plan C" (true in-scene parallel rendering).

Usage:
    python tools/parallel_render_poc.py
    python tools/parallel_render_poc.py --workers 1 2 4 8 --frames 20
"""
from __future__ import annotations

import argparse
import ctypes
import json
import math
import multiprocessing as mp
import os
import struct
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(r"D:\RTXNS")
SCENE_DIR = Path(r"D:\niagara_bistro")
OUT_DIR = REPO_ROOT / "output" / "parallel_render"


# --------------------------------------------------------------------------- #
# Niagara camera loading (copied from test_bistro_shadow.py)
# --------------------------------------------------------------------------- #
class _NH(ctypes.Structure):
    _fields_ = [
        ("mag", ctypes.c_uint32), ("ver", ctypes.c_uint32),
        ("mmv", ctypes.c_uint32), ("mmt", ctypes.c_uint32),
        ("clrt", ctypes.c_bool), ("cmp", ctypes.c_bool),
        ("cvb", ctypes.c_uint32), ("cib", ctypes.c_uint32),
        ("cmd", ctypes.c_uint32), ("cmv", ctypes.c_uint32),
        ("vc", ctypes.c_uint32), ("ic", ctypes.c_uint32),
        ("mc", ctypes.c_uint32), ("mdc", ctypes.c_uint32),
        ("mvc", ctypes.c_uint32), ("meshc", ctypes.c_uint32),
        ("matc", ctypes.c_uint32), ("dc", ctypes.c_uint32),
        ("tpc", ctypes.c_uint32), ("oads", ctypes.c_uint32),
        ("oids", ctypes.c_uint32), ("odc", ctypes.c_uint32),
        ("oms", ctypes.c_uint32),
        ("cam_p", ctypes.c_float * 3),
        ("cam_o", ctypes.c_float * 4),
        ("cam_f", ctypes.c_float),
        ("cam_z", ctypes.c_float),
        ("sd", ctypes.c_float * 3),
    ]


def _rotate_by_quat_xyzw(q, v):
    x, y, z, w = q
    cx = y * v[2] - z * v[1]
    cy = z * v[0] - x * v[2]
    cz = x * v[1] - y * v[0]
    tx, ty, tz = 2.0 * cx, 2.0 * cy, 2.0 * cz
    c2x = y * tz - z * ty
    c2y = z * tx - x * tz
    c2z = x * ty - y * tx
    return [v[0] + w * tx + c2x, v[1] + w * ty + c2y, v[2] + w * tz + c2z]


def load_niagara_inputs(scene_dir: Path):
    hb = (scene_dir / "bistro.gltf.cache").read_bytes()[:ctypes.sizeof(_NH)]
    h = _NH.from_buffer_copy(hb)
    cb = (scene_dir / "bistro.gltf.camera").read_bytes()
    cv, px, py, pz, qx, qy, qz, qw = struct.unpack("<I3f4f", cb)
    pos = [px, py, pz]
    quat = [qx, qy, qz, qw]
    fw = _rotate_by_quat_xyzw(quat, [0, 0, -1])
    up = _rotate_by_quat_xyzw(quat, [0, 1, 0])
    tgt = [pos[i] + fw[i] for i in range(3)]
    fov = float(h.cam_f) * 180 / math.pi
    zn = float(h.cam_z)
    sun = [float(v) for v in h.sd]
    return pos, tgt, up, fov, zn, sun


def orbit_camera(base_pos, base_target, angle_deg: float):
    """Orbit the camera around base_target by angle_deg (yaw)."""
    a = math.radians(angle_deg)
    dx = base_pos[0] - base_target[0]
    dz = base_pos[2] - base_target[2]
    nx = dx * math.cos(a) - dz * math.sin(a)
    nz = dx * math.sin(a) + dz * math.cos(a)
    new_pos = [base_target[0] + nx, base_pos[1], base_target[2] + nz]
    return new_pos, base_target


# --------------------------------------------------------------------------- #
# Worker process
# --------------------------------------------------------------------------- #
@dataclass
class WorkerResult:
    worker_id: int
    angle_deg: float
    num_frames: int
    steady_fps: float
    total_wall_s: float
    mean_pixel: float
    frame_times_ms: List[float] = field(default_factory=list)
    error: str = ""


def _worker_main(
    worker_id: int,
    angle_deg: float,
    num_frames: int,
    out_dir: str,
    result_queue: mp.Queue,
):
    """Entry point for each worker process."""
    try:
        # IMPORTANT: import inside worker so each process gets its own module state
        sys.path.insert(0, str(REPO_ROOT / "bin" / "windows-x64"))
        import DonutRenderPyNative as rr

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        W, H = 1024, 768

        rr.init(runtime_dir=str(REPO_ROOT), backend="vulkan",
                device_index=-1, enable_debug=False)
        scene = rr.create_scene()

        base_pos, base_tgt, up, fov, zn, sun = load_niagara_inputs(SCENE_DIR)
        cam_pos, cam_tgt = orbit_camera(base_pos, base_tgt, angle_deg)

        t_load0 = time.time()
        scene.load_scene(str(SCENE_DIR / "bistro.gltf"))
        t_load = time.time() - t_load0

        scene.set_camera(position=cam_pos, target=cam_tgt, up=up,
                         fov_degrees=fov, width=W, height=H,
                         z_near=zn, z_far=200.0)
        scene.set_default_light(direction=[-v for v in sun],
                                color=[1, 1, 1], irradiance=50)
        scene.set_ambient(top_rgb=[0.03, 0.03, 0.03],
                          bottom_rgb=[0.01, 0.01, 0.01])
        scene.enable_rt_shadows(True)
        scene.enable_shadow_blur(True)
        scene.set_shadow_samples(8)

        # Warm up (BLAS build, first frame)
        img0 = scene.render_frame()
        st0 = scene.get_last_frame_stats()

        # Save first frame image for visual confirmation
        arr0 = np.frombuffer(img0, np.uint8).reshape(H, W, 4)[:, :, :3]
        Image.fromarray(arr0, "RGB").save(
            OUT_DIR / f"worker_{worker_id}_angle{int(angle_deg)}.png")

        # Steady-state benchmark
        frame_times = []
        imgs = []
        t_start = time.time()
        for _ in range(num_frames):
            t0 = time.time()
            img = scene.render_frame()
            t1 = time.time()
            frame_times.append((t1 - t0) * 1000.0)
            imgs.append(img)
        t_end = time.time()

        steady_wall = t_end - t_start
        steady_fps = num_frames / steady_wall if steady_wall > 0 else 0.0

        # mean pixel of last frame
        arr_last = np.frombuffer(imgs[-1], np.uint8).reshape(H, W, 4)[:, :, :3]
        mean_px = float(arr_last.mean())

        rr.destroy()

        result = WorkerResult(
            worker_id=worker_id,
            angle_deg=angle_deg,
            num_frames=num_frames,
            steady_fps=steady_fps,
            total_wall_s=steady_wall,
            mean_pixel=mean_px,
            frame_times_ms=frame_times,
        )
        result_queue.put(result)

    except Exception as e:
        import traceback
        err = traceback.format_exc()
        result_queue.put(WorkerResult(
            worker_id=worker_id, angle_deg=angle_deg, num_frames=num_frames,
            steady_fps=0.0, total_wall_s=0.0, mean_pixel=0.0, error=str(err)))
        print(f"[Worker {worker_id}] ERROR:\n{err}", file=sys.stderr)


# --------------------------------------------------------------------------- #
# Benchmark runner
# --------------------------------------------------------------------------- #
def run_benchmark(num_workers: int, num_frames: int) -> Tuple[float, List[WorkerResult]]:
    """Run N worker processes in parallel, return (wall_clock_s, results)."""
    # Distribute camera angles evenly around the target
    angles = [i * (360.0 / num_workers) for i in range(num_workers)]

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    procs = []
    t_wall_start = time.time()
    for i, ang in enumerate(angles):
        p = ctx.Process(target=_worker_main,
                        args=(i, ang, num_frames, str(OUT_DIR), result_queue))
        p.start()
        procs.append(p)

    # Wait for all
    for p in procs:
        p.join()

    t_wall_end = time.time()
    wall = t_wall_end - t_wall_start

    results = []
    while not result_queue.empty():
        results.append(result_queue.get_nowait())
    results.sort(key=lambda r: r.worker_id)
    return wall, results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4],
                        help="List of worker counts to benchmark")
    parser.add_argument("--frames", type=int, default=15,
                        help="Number of steady-state frames per worker")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_runs = []
    print(f"=== Parallel Rendering POC ===")
    print(f"Scene: Bistro 1024x768, RT shadow + 8-sample blur")
    print(f"Frames per worker (steady-state): {args.frames}")
    print(f"Worker counts: {args.workers}")
    print()

    for n in args.workers:
        print(f"--- Running N={n} worker(s) ---")
        wall, results = run_benchmark(n, args.frames)
        total_frames = sum(r.num_frames for r in results if not r.error)
        avg_worker_fps = (sum(r.steady_fps for r in results if not r.error)
                          / max(1, sum(1 for r in results if not r.error)))
        # Steady-state throughput = sum of all workers' FPS (excludes scene load)
        steady_throughput = sum(r.steady_fps for r in results if not r.error)
        # Wall-clock throughput includes scene load + BLAS build (real-world)
        wall_throughput = total_frames / wall if wall > 0 else 0.0

        run = {
            "num_workers": n,
            "wall_clock_s": wall,
            "total_frames": total_frames,
            "steady_throughput_fps": steady_throughput,
            "wall_throughput_fps": wall_throughput,
            "avg_worker_fps": avg_worker_fps,
            "results": [
                {"worker_id": r.worker_id, "angle": r.angle_deg,
                 "fps": r.steady_fps, "mean_pixel": r.mean_pixel,
                 "frame_times_ms": r.frame_times_ms, "error": r.error}
                for r in results
            ],
        }
        all_runs.append(run)

        for r in results:
            if r.error:
                print(f"  Worker {r.worker_id} (angle {r.angle_deg:.0f}°): ERROR")
                print(f"    {r.error}")
            else:
                print(f"  Worker {r.worker_id} (angle {r.angle_deg:.0f}°): "
                      f"{r.steady_fps:.1f} FPS, mean={r.mean_pixel:.1f}")
        print(f"  Wall clock: {wall:.2f}s (incl. load), "
              f"Steady throughput: {steady_throughput:.1f} FPS, "
              f"Wall throughput: {wall_throughput:.1f} FPS")
        print()

    # Save JSON
    json_path = OUT_DIR / "parallel_render_results.json"
    with open(json_path, "w") as f:
        json.dump(all_runs, f, indent=2)
    print(f"Saved results to {json_path}")

    # Generate throughput comparison chart
    _make_chart(all_runs)


def _make_chart(all_runs):
    """Bar chart: throughput vs num_workers, with ideal linear scaling line."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ns = [r["num_workers"] for r in all_runs]
    throughputs = [r["steady_throughput_fps"] for r in all_runs]
    avg_worker = [r["avg_worker_fps"] for r in all_runs]
    wall_thru = [r["wall_throughput_fps"] for r in all_runs]

    baseline = throughputs[0] if throughputs else 1.0
    ideal = [baseline * n for n in ns]
    speedups = [t / baseline for t in throughputs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: throughput
    x = np.arange(len(ns))
    w = 0.35
    ax1.bar(x - w/2, throughputs, w, label="Actual throughput", color="#4a90d9")
    ax1.bar(x + w/2, ideal, w, label="Ideal (linear)", color="#d9d9d9",
            hatch="//", edgecolor="#888")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"N={n}" for n in ns])
    ax1.set_ylabel("Steady-state Throughput (FPS)")
    ax1.set_title("Parallel Rendering Throughput\n(Bistro 1024×768, RT shadow + 8-sample, excl. load)")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)
    for i, (t, s) in enumerate(zip(throughputs, speedups)):
        ax1.text(i - w/2, t + 1, f"{t:.1f}\n({s:.2f}x)",
                 ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Right: per-worker FPS degradation
    ax2.plot(ns, avg_worker, "o-", color="#e74c3c", linewidth=2, markersize=8)
    ax2.axhline(y=baseline, color="#888", linestyle="--", label="Baseline (N=1)")
    ax2.set_xlabel("Number of worker processes")
    ax2.set_ylabel("Avg per-worker FPS")
    ax2.set_title("Per-Worker FPS vs Parallelism\n(GPU contention indicator)")
    ax2.legend()
    ax2.grid(alpha=0.3)
    for i, (n, fps) in enumerate(zip(ns, avg_worker)):
        ax2.text(n, fps + 2, f"{fps:.1f}", ha="center", fontsize=9)

    plt.tight_layout()
    chart_path = OUT_DIR / "throughput_compare.png"
    plt.savefig(chart_path, dpi=120)
    print(f"Saved chart to {chart_path}")


if __name__ == "__main__":
    main()
