"""Diagnostic: genuine OS-process concurrency for the ReplicaCAD RT workload.

This is deliberately separate from the shared-scene async batch benchmark.
Each worker owns a renderer and scene, so it can reveal the real throughput
available from concurrent Vulkan submissions, but it cannot claim the batch
path's shared-memory efficiency.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import statistics
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_DIR = REPO_ROOT / "bin" / "windows-x64"
SCENE_PATH = REPO_ROOT / "data" / "replica_cad" / "frl_apartment_stage.glb"
OUT_DIR = REPO_ROOT / "output" / "replicacad_multiprocess_rt"
RING_DEPTH = 4
CAMERA = dict(
    position=(3.5, 2.0, 3.5), target=(0.0, 1.0, 0.0), up=(0.0, 1.0, 0.0),
    fov_degrees=60.0, width=1280, height=720, z_near=0.1, z_far=100.0,
)


def query_vram_mb() -> int:
    output = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout
    return int(output.splitlines()[0].strip())


class Sampler:
    def __init__(self):
        self.values: list[int] = []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self.stop_event.is_set():
            try:
                self.values.append(query_vram_mb())
            except (OSError, subprocess.SubprocessError, ValueError):
                pass
            self.stop_event.wait(0.2)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=3)


def worker_main(worker_id: int, frames: int, ready_queue, start_event, result_queue) -> None:
    """One independent renderer: this is real process concurrency, not batch."""
    try:
        sys.path.insert(0, str(MODULE_DIR))
        import DonutRenderPyNative as rr

        rr.init(runtime_dir=str(REPO_ROOT), backend="vulkan", device_index=-1, enable_debug=False)
        scene = rr.create_scene()
        scene.load_scene(str(SCENE_PATH))
        scene.set_ambient((0.03, 0.04, 0.06), (0.01, 0.01, 0.01))
        scene.set_default_light((-0.4, -1.0, -0.6), (1.0, 1.0, 1.0), 2.0)
        scene.enable_rt_shadows(True)
        scene.enable_shadow_blur(True)
        scene.set_shadow_samples(8)
        scene.set_readback_ring_depth(RING_DEPTH)
        camera_id = scene.add_camera(**CAMERA)
        # Warmup the same async batch path used by the current renderer.  A
        # process still owns one independent scene/device; the EventQuery and
        # ring remove the worker's per-frame idle wait from steady state.
        scene.render_frame_batch([camera_id])

        ready_queue.put((worker_id, "ready", ""))
        if not start_event.wait(timeout=90):
            raise TimeoutError("benchmark start event timed out")

        pending: list[int] = []
        submitted = 0
        start = time.perf_counter()
        while submitted < frames:
            token = scene.submit_frame_batch([camera_id])
            if token:
                pending.append(token)
                submitted += 1
                if len(pending) >= RING_DEPTH:
                    scene.read_frame_batch(pending.pop(0))
            elif pending:
                scene.read_frame_batch(pending.pop(0))
        for token in pending:
            scene.read_frame_batch(token)
        elapsed = time.perf_counter() - start
        rr.destroy()
        result_queue.put({"worker_id": worker_id, "fps": frames / elapsed, "error": ""})
    except Exception:
        error = traceback.format_exc()
        try:
            ready_queue.put((worker_id, "error", error))
        except Exception:
            pass
        result_queue.put({"worker_id": worker_id, "fps": 0.0, "error": error})


def run_case(workers: int, frames: int) -> dict:
    # Device-wide accounting is reported as a delta because the desktop has
    # unrelated graphics clients.  Worker processes are aligned after warmup
    # so this captures all independent scene allocations simultaneously.
    baseline_values = [query_vram_mb() for _ in range(3)]
    baseline = int(round(statistics.median(baseline_values)))
    ctx = mp.get_context("spawn")
    ready_queue = ctx.Queue()
    result_queue = ctx.Queue()
    start_event = ctx.Event()
    sampler = Sampler()
    sampler.start()
    processes = [ctx.Process(target=worker_main, args=(i, frames, ready_queue, start_event, result_queue))
                 for i in range(workers)]
    for process in processes:
        process.start()

    ready = 0
    errors: list[str] = []
    deadline = time.monotonic() + 120
    while ready < workers and time.monotonic() < deadline:
        try:
            _, state, message = ready_queue.get(timeout=1)
            if state == "ready":
                ready += 1
            else:
                errors.append(message)
                break
        except Exception:
            pass

    # Hold after every worker has loaded to sample resident allocations.
    time.sleep(0.8)
    resident = query_vram_mb()
    if ready == workers and not errors:
        start_event.set()
    else:
        errors.append("not every worker reached the warmup barrier")
        start_event.set()

    for process in processes:
        process.join(timeout=120)
        if process.is_alive():
            process.terminate()
            errors.append(f"worker PID {process.pid} timed out")
    sampler.stop()
    results = []
    while not result_queue.empty():
        results.append(result_queue.get_nowait())
    errors.extend(r["error"] for r in results if r["error"])
    ok = [r for r in results if not r["error"]]
    peak = max(sampler.values, default=resident)
    return {
        "workers": workers,
        "throughput_fps": sum(r["fps"] for r in ok),
        "avg_worker_fps": (sum(r["fps"] for r in ok) / len(ok)) if ok else 0.0,
        "baseline_vram_mb": baseline,
        "resident_vram_mb": resident,
        "peak_vram_mb": peak,
        "resident_delta_mb": max(0, resident - baseline),
        "peak_delta_mb": max(0, peak - baseline),
        "results": results,
        "errors": errors,
    }


def make_chart(results: list[dict]) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ns = [r["workers"] for r in results]
    throughput = [r["throughput_fps"] for r in results]
    throughput_std = [r["throughput_fps_std"] for r in results]
    resident = [r["resident_delta_mb"] for r in results]
    peak = [r["peak_delta_mb"] for r in results]
    base = throughput[0]
    ideal = [base * n for n in ns]
    speedup = [value / base for value in throughput]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(ns))
    width = 0.35
    ax1.bar(x - width / 2, throughput, width, yerr=throughput_std, capsize=3,
            color="#4a90d9", label="Async multi-process throughput")
    ax1.bar(x + width / 2, ideal, width, color="#d9d9d9", hatch="//",
            edgecolor="#888", label="Ideal (linear)")
    ax1.set_xticks(x, [f"N={n}" for n in ns])
    ax1.set_ylabel("Steady-state throughput (FPS)")
    ax1.set_title("ReplicaCAD Async Worker-Pool Throughput\n(RT shadow 8-sample, no GI, .60 HBD, end-to-end)")
    ax1.grid(axis="y", alpha=0.3)
    ax1.legend(loc="upper left")
    for i, (value, ratio) in enumerate(zip(throughput, speedup)):
        ax1.text(i - width / 2, value + max(throughput) * 0.015,
                 f"{value:.0f}\n({ratio:.2f}x)", ha="center", va="bottom",
                 fontsize=9, fontweight="bold")

    ax2.bar(ns, resident, color="#27ae60", alpha=0.85, label="Resident ΔVRAM")
    ax2.scatter(ns, peak, color="#e74c3c", zorder=3, label="Peak ΔVRAM")
    ax2.set_xticks(ns)
    ax2.set_xlabel("Number of independent renderer workers")
    ax2.set_ylabel("Additional GPU memory vs idle (MB)")
    ax2.set_title("ReplicaCAD Worker-Pool Memory Footprint\n(each worker owns renderer / scene / AS)")
    ax2.grid(axis="y", alpha=0.3)
    ax2.legend(loc="upper left")
    ax2.set_ylim(0, max(peak) * 1.22)
    for n, resident_mb, peak_mb in zip(ns, resident, peak):
        ax2.text(n, peak_mb + max(30, max(peak) * 0.025),
                 f"{resident_mb} / {peak_mb} MB\n({100 * peak_mb / 16303:.1f}% of 16GB)",
                 ha="center", fontsize=8)
    ax2.text(0.02, 0.02, "labels: resident / peak ΔVRAM; medians of 3 trials",
             transform=ax2.transAxes, fontsize=8, color="#555")

    fig.suptitle("Flora ReplicaCAD Parallel Rendering — Async Worker Pool",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = OUT_DIR / "replicacad_async_worker_throughput_vram.png"
    fig.savefig(path, dpi=140)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4, 6, 8])
    parser.add_argument("--frames", type=int, default=90)
    parser.add_argument("--trials", type=int, default=3)
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    print("ReplicaCAD multi-process async diagnostic: RT shadow=8, GI=off, HBD=.60, ring K=4")
    for workers in args.workers:
        print(f"--- N={workers} independent renderers ---", flush=True)
        trials = []
        for trial in range(args.trials):
            result = run_case(workers, args.frames)
            trials.append(result)
            print(f"trial {trial + 1}/{args.trials}: throughput={result['throughput_fps']:.1f} FPS, "
                  f"VRAM resident/peak Δ={result['resident_delta_mb']}/{result['peak_delta_mb']}MB, "
                  f"errors={len(result['errors'])}", flush=True)
        aggregate = {
            "workers": workers,
            "throughput_fps": float(np.mean([r["throughput_fps"] for r in trials])),
            "throughput_fps_std": float(np.std([r["throughput_fps"] for r in trials], ddof=0)),
            "avg_worker_fps": float(np.mean([r["avg_worker_fps"] for r in trials])),
            "resident_delta_mb": int(round(statistics.median(r["resident_delta_mb"] for r in trials))),
            "peak_delta_mb": int(round(statistics.median(r["peak_delta_mb"] for r in trials))),
            "trials": trials,
        }
        results.append(aggregate)
        print(f"mean: throughput={aggregate['throughput_fps']:.1f}±{aggregate['throughput_fps_std']:.1f} FPS, "
              f"VRAM resident/peak Δ={aggregate['resident_delta_mb']}/{aggregate['peak_delta_mb']}MB", flush=True)
    base = results[0]["throughput_fps"]
    for result in results:
        result["speedup"] = result["throughput_fps"] / base
    json_path = OUT_DIR / "multiprocess_async_results.json"
    json_path.write_text(
        json.dumps(results, indent=2), encoding="utf-8")
    chart_path = make_chart(results)
    print(f"Saved: {json_path}")
    print(f"Saved: {chart_path}")


if __name__ == "__main__":
    main()
