"""Parallel rendering bottleneck analysis — find where parallelism stops scaling.

Extends parallel_render_poc.py with:
  1. GPU utilization sampling (nvidia-smi background poller)
  2. VRAM usage tracking (per-N)
  3. CPU-side overhead: render_frame() wall time vs GPU total_ms
  4. Extended N sweep: 1/2/4/6/8 to find the scaling knee

Output: 4-panel chart isolating the bottleneck that justifies Plan C.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import math
import multiprocessing as mp
import re
import struct
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

REPO_ROOT = Path(r"D:\RTXNS")
SCENE_DIR = Path(r"D:\niagara_bistro")
OUT_DIR = REPO_ROOT / "output" / "parallel_render"


# --------------------------------------------------------------------------- #
# Niagara camera loading (shared with parallel_render_poc.py)
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
    a = math.radians(angle_deg)
    dx = base_pos[0] - base_target[0]
    dz = base_pos[2] - base_target[2]
    nx = dx * math.cos(a) - dz * math.sin(a)
    nz = dx * math.sin(a) + dz * math.cos(a)
    new_pos = [base_target[0] + nx, base_pos[1], base_target[2] + nz]
    return new_pos, base_target


# --------------------------------------------------------------------------- #
# GPU monitor (background nvidia-smi poller in main process)
# --------------------------------------------------------------------------- #
class GPUMonitor:
    """Background nvidia-smi sampler. Call .start()/.stop(), read .samples."""

    def __init__(self, interval_s: float = 0.3):
        self.interval_s = interval_s
        self.samples: List[Tuple[float, int, int]] = []  # (t, gpu_util%, vram_mb)
        self._proc = None
        self._t0 = 0.0

    def start(self):
        self._t0 = time.time()
        cmd = [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
            "-l", str(self.interval_s),
        ]
        self._proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, encoding="utf-8", errors="replace")

    def stop(self):
        if self._proc is None:
            return
        try:
            self._proc.terminate()
            out, _ = self._proc.communicate(timeout=2)
        except Exception:
            out = ""
            try:
                self._proc.kill()
            except Exception:
                pass
        for line in out.splitlines():
            line = line.strip()
            if not line:
                continue
            m = re.match(r"\s*(\d+)\s*,\s*(\d+)", line)
            if m:
                util = int(m.group(1))
                vram = int(m.group(2))
                self.samples.append((time.time() - self._t0, util, vram))

    def stats(self):
        if not self.samples:
            return {"avg_util": 0, "peak_util": 0, "avg_vram_mb": 0, "peak_vram_mb": 0}
        utils = [s[1] for s in self.samples]
        vrams = [s[2] for s in self.samples]
        return {
            "avg_util": float(np.mean(utils)),
            "peak_util": int(np.max(utils)),
            "avg_vram_mb": float(np.mean(vrams)),
            "peak_vram_mb": int(np.max(vrams)),
            "num_samples": len(self.samples),
        }


# --------------------------------------------------------------------------- #
# Worker
# --------------------------------------------------------------------------- #
@dataclass
class WorkerResult:
    worker_id: int
    angle_deg: float
    num_frames: int
    steady_fps: float
    steady_throughput: float  # sum of all workers
    avg_call_wall_ms: float   # render_frame() wall time
    avg_gpu_total_ms: float   # from get_last_frame_stats
    avg_cpu_overhead_ms: float  # call_wall - gpu_total
    mean_pixel: float
    frame_call_walls_ms: List[float] = field(default_factory=list)
    frame_gpu_totals_ms: List[float] = field(default_factory=list)
    error: str = ""


def _worker_main(
    worker_id: int,
    angle_deg: float,
    num_frames: int,
    out_dir: str,
    result_queue: mp.Queue,
):
    try:
        sys.path.insert(0, str(REPO_ROOT / "bin" / "windows-x64"))
        import DonutRenderPyNative as rr

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        W, H = 1024, 768

        rr.init(runtime_dir=str(REPO_ROOT), backend="vulkan",
                device_index=-1, enable_debug=False)
        scene = rr.create_scene()

        base_pos, base_tgt, up, fov, zn, sun = load_niagara_inputs(SCENE_DIR)
        cam_pos, cam_tgt = orbit_camera(base_pos, base_tgt, angle_deg)

        scene.load_scene(str(SCENE_DIR / "bistro.gltf"))
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

        # Warmup
        scene.render_frame()

        # Steady-state: measure both wall and GPU-reported time
        call_walls = []
        gpu_totals = []
        imgs = []
        t_start = time.time()
        for _ in range(num_frames):
            t0 = time.time()
            img = scene.render_frame()
            t1 = time.time()
            st = scene.get_last_frame_stats()
            call_walls.append((t1 - t0) * 1000.0)
            gpu_totals.append(float(st.get("total_ms", 0.0)))
            imgs.append(img)
        t_end = time.time()

        steady_fps = num_frames / (t_end - t_start) if t_end > t_start else 0.0
        avg_call_wall = float(np.mean(call_walls))
        avg_gpu_total = float(np.mean(gpu_totals))
        avg_cpu_overhead = max(0.0, avg_call_wall - avg_gpu_total)

        arr_last = np.frombuffer(imgs[-1], np.uint8).reshape(H, W, 4)[:, :, :3]
        rr.destroy()

        result_queue.put(WorkerResult(
            worker_id=worker_id,
            angle_deg=angle_deg,
            num_frames=num_frames,
            steady_fps=steady_fps,
            steady_throughput=0.0,  # filled by main
            avg_call_wall_ms=avg_call_wall,
            avg_gpu_total_ms=avg_gpu_total,
            avg_cpu_overhead_ms=avg_cpu_overhead,
            mean_pixel=float(arr_last.mean()),
            frame_call_walls_ms=call_walls,
            frame_gpu_totals_ms=gpu_totals,
        ))

    except Exception as e:
        import traceback
        err = traceback.format_exc()
        result_queue.put(WorkerResult(
            worker_id=worker_id, angle_deg=angle_deg, num_frames=num_frames,
            steady_fps=0.0, steady_throughput=0.0,
            avg_call_wall_ms=0.0, avg_gpu_total_ms=0.0,
            avg_cpu_overhead_ms=0.0, mean_pixel=0.0, error=str(err)))
        print(f"[Worker {worker_id}] ERROR:\n{err}", file=sys.stderr)


# --------------------------------------------------------------------------- #
# Benchmark runner with GPU monitoring
# --------------------------------------------------------------------------- #
def run_benchmark(num_workers: int, num_frames: int):
    angles = [i * (360.0 / num_workers) for i in range(num_workers)]
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    procs = []

    monitor = GPUMonitor(interval_s=0.3)
    monitor.start()
    t_wall_start = time.time()

    for i, ang in enumerate(angles):
        p = ctx.Process(target=_worker_main,
                        args=(i, ang, num_frames, str(OUT_DIR), result_queue))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    t_wall_end = time.time()
    monitor.stop()

    wall = t_wall_end - t_wall_start
    results = []
    while not result_queue.empty():
        results.append(result_queue.get_nowait())
    results.sort(key=lambda r: r.worker_id)

    # Fill in aggregate
    steady_throughput = sum(r.steady_fps for r in results if not r.error)
    for r in results:
        r.steady_throughput = steady_throughput

    return wall, results, monitor.stats()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, nargs="+",
                        default=[1, 2, 4, 6, 8])
    parser.add_argument("--frames", type=int, default=15)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_runs = []
    print(f"=== Parallel Rendering Bottleneck Analysis ===")
    print(f"Scene: Bistro 1024x768, RT shadow + 8-sample")
    print(f"Frames per worker: {args.frames}")
    print(f"Worker counts: {args.workers}")
    print()

    for n in args.workers:
        print(f"--- N={n} ---")
        try:
            wall, results, gpu_stats = run_benchmark(n, args.frames)
        except Exception as e:
            print(f"  FAILED: {e}")
            all_runs.append({
                "num_workers": n, "error": str(e),
                "steady_throughput_fps": 0, "avg_worker_fps": 0,
                "speedup": 0, "gpu_stats": {}, "results": [],
            })
            continue

        ok = [r for r in results if not r.error]
        if not ok:
            print(f"  All workers failed")
            for r in results:
                print(f"    W{r.worker_id}: {r.error[:200]}")
            continue

        steady_thru = sum(r.steady_fps for r in ok)
        avg_fps = steady_thru / len(ok)
        avg_call = float(np.mean([r.avg_call_wall_ms for r in ok]))
        avg_gpu = float(np.mean([r.avg_gpu_total_ms for r in ok]))
        avg_cpu_oh = float(np.mean([r.avg_cpu_overhead_ms for r in ok]))

        run = {
            "num_workers": n,
            "wall_clock_s": wall,
            "steady_throughput_fps": steady_thru,
            "avg_worker_fps": avg_fps,
            "avg_call_wall_ms": avg_call,
            "avg_gpu_total_ms": avg_gpu,
            "avg_cpu_overhead_ms": avg_cpu_oh,
            "gpu_stats": gpu_stats,
            "results": [
                {"worker_id": r.worker_id, "angle": r.angle_deg,
                 "fps": r.steady_fps, "call_wall_ms": r.avg_call_wall_ms,
                 "gpu_total_ms": r.avg_gpu_total_ms,
                 "cpu_overhead_ms": r.avg_cpu_overhead_ms,
                 "mean_pixel": r.mean_pixel, "error": r.error}
                for r in results
            ],
        }
        all_runs.append(run)

        for r in ok:
            print(f"  W{r.worker_id} ({r.angle_deg:.0f}°): "
                  f"{r.steady_fps:.1f} FPS, "
                  f"call={r.avg_call_wall_ms:.2f}ms, "
                  f"gpu={r.avg_gpu_total_ms:.2f}ms, "
                  f"cpu_oh={r.avg_cpu_overhead_ms:.2f}ms")
        print(f"  Steady throughput: {steady_thru:.1f} FPS")
        print(f"  GPU util: avg={gpu_stats['avg_util']:.0f}% "
              f"peak={gpu_stats['peak_util']}%, "
              f"VRAM: avg={gpu_stats['avg_vram_mb']:.0f}MB "
              f"peak={gpu_stats['peak_vram_mb']}MB")
        print()

    # Speedup relative to N=1
    base = all_runs[0]["steady_throughput_fps"] if all_runs else 1.0
    for r in all_runs:
        r["speedup"] = r["steady_throughput_fps"] / base if base > 0 else 0.0
        print(f"N={r['num_workers']}: throughput={r['steady_throughput_fps']:.1f} "
              f"FPS, speedup={r['speedup']:.2f}x, "
              f"GPU={r.get('gpu_stats', {}).get('avg_util', 0):.0f}%")

    json_path = OUT_DIR / "bottleneck_results.json"
    with open(json_path, "w") as f:
        json.dump(all_runs, f, indent=2)
    print(f"\nSaved results to {json_path}")

    _make_chart(all_runs)


def _make_chart(all_runs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valid = [r for r in all_runs if not r.get("error") and r["steady_throughput_fps"] > 0]
    if not valid:
        print("No valid runs to chart")
        return

    ns = [r["num_workers"] for r in valid]
    throughputs = [r["steady_throughput_fps"] for r in valid]
    speedups = [r["speedup"] for r in valid]
    gpu_utils = [r.get("gpu_stats", {}).get("avg_util", 0) for r in valid]
    vrams = [r.get("gpu_stats", {}).get("peak_vram_mb", 0) for r in valid]
    cpu_overheads = [r.get("avg_cpu_overhead_ms", 0) for r in valid]
    avg_worker_fps = [r.get("avg_worker_fps", 0) for r in valid]

    baseline = throughputs[0] if throughputs else 1.0
    ideal_speedup = [1.0 * n for n in ns]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # (1) Throughput + speedup
    ax = axes[0, 0]
    x = np.arange(len(ns))
    w = 0.35
    ax.bar(x - w/2, throughputs, w, label="Actual throughput", color="#4a90d9")
    ax.bar(x + w/2, [baseline * n for n in ns], w,
           label="Ideal (linear)", color="#d9d9d9", hatch="//", edgecolor="#888")
    ax.set_xticks(x)
    ax.set_xticklabels([f"N={n}" for n in ns])
    ax.set_ylabel("Steady-state Throughput (FPS)")
    ax.set_title("(1) Throughput & Scaling Knee")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for i, (t, s) in enumerate(zip(throughputs, speedups)):
        ax.text(i - w/2, t + 20, f"{t:.0f}\n({s:.2f}x)",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    # (2) GPU utilization
    ax = axes[0, 1]
    ax.plot(ns, gpu_utils, "o-", color="#e74c3c", linewidth=2, markersize=8,
            label="Avg GPU util")
    ax.axhline(y=100, color="#888", linestyle="--", label="100% (saturated)")
    ax.axhline(y=80, color="#aaa", linestyle=":", label="80% (near-saturation)")
    ax.set_xlabel("Number of worker processes")
    ax.set_ylabel("GPU Utilization (%)")
    ax.set_title("(2) GPU Utilization vs Parallelism\n(Bottleneck: is GPU saturated?)")
    ax.legend()
    ax.grid(alpha=0.3)
    for i, (n, u) in enumerate(zip(ns, gpu_utils)):
        ax.text(n, u + 2, f"{u:.0f}%", ha="center", fontsize=9)

    # (3) VRAM usage
    ax = axes[1, 0]
    ax.bar(ns, vrams, color="#27ae60", alpha=0.8, label="Peak VRAM")
    ax.axhline(y=32768, color="#e74c3c", linestyle="--",
               label="RTX 5090D limit (32GB)")
    ax.set_xlabel("Number of worker processes")
    ax.set_ylabel("VRAM Usage (MB)")
    ax.set_title("(3) VRAM Scaling — Multi-Process Overhead\n(Plan C can fix: shared device)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for i, (n, v) in enumerate(zip(ns, vrams)):
        ax.text(n, v + 200, f"{v}\n({100*v/32768:.0f}%)",
                ha="center", fontsize=9)

    # (4) CPU overhead per frame
    ax = axes[1, 1]
    ax.plot(ns, cpu_overheads, "s-", color="#8e44ad", linewidth=2,
            markersize=8, label="CPU overhead / frame")
    ax.set_xlabel("Number of worker processes")
    ax.set_ylabel("CPU Overhead (ms / frame / worker)")
    ax.set_title("(4) CPU-Side Overhead per Frame\n(call_wall - gpu_total; Plan C fixes via batch API)")
    ax.legend()
    ax.grid(alpha=0.3)
    for i, (n, c) in enumerate(zip(ns, cpu_overheads)):
        ax.text(n, c + 0.1, f"{c:.2f}ms", ha="center", fontsize=9)

    plt.suptitle("Flora Parallel Rendering — Bottleneck Analysis\n"
                 "(Bistro 1024×768, FloraRT shadow + 8-sample, RTX 5090D 32GB)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    chart_path = OUT_DIR / "bottleneck_analysis.png"
    plt.savefig(chart_path, dpi=120)
    print(f"Saved chart to {chart_path}")


if __name__ == "__main__":
    main()
