"""ReplicaCAD async-batch scaling benchmark with a common presentation path.

Measures the current single-renderer / shared-scene camera batch path, rather
than the superseded one-process-per-camera POC.  Every case uses the same
ReplicaCAD apartment, camera pose, FloraRT 8-sample shadow, no GI, and the
global 0.60 HBD display transform configured in shadow_composite_cs.hlsl.

The N cameras intentionally share the same pose.  This isolates camera-count
scaling and view/render-target memory from scene-visibility differences.

Usage (run with the Python environment that can import NumPy/Pillow):
    python tools/bench_replicacad_parallel_common_tonemap.py
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_DIR = REPO_ROOT / "bin" / "windows-x64"
SCENE_PATH = REPO_ROOT / "data" / "replica_cad" / "frl_apartment_stage.glb"
OUT_DIR = REPO_ROOT / "output" / "replicacad_parallel_common_tonemap"

WIDTH, HEIGHT = 1280, 720
RING_DEPTH = 4
CAMERA = dict(
    position=(3.5, 2.0, 3.5),
    target=(0.0, 1.0, 0.0),
    up=(0.0, 1.0, 0.0),
    fov_degrees=60.0,
    width=WIDTH,
    height=HEIGHT,
    z_near=0.1,
    z_far=100.0,
)


def query_vram_mb() -> int:
    """Return device-wide used VRAM in MB from nvidia-smi."""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    )
    return int(result.stdout.splitlines()[0].strip())


def median_vram(samples: int = 3, interval_s: float = 0.2) -> int:
    values = []
    for _ in range(samples):
        values.append(query_vram_mb())
        time.sleep(interval_s)
    return int(round(statistics.median(values)))


class VramSampler:
    """Sample total GPU usage while a benchmark case is resident."""

    def __init__(self, interval_s: float = 0.2):
        self.interval_s = interval_s
        self.values: list[int] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.values.append(query_vram_mb())
            except (OSError, subprocess.SubprocessError, ValueError):
                pass
            self._stop.wait(self.interval_s)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=3)


def async_e2e_benchmark(scene, camera_ids: list[int], batches: int) -> tuple[float, float, int]:
    """Return camera FPS, batch ms, and ring-backpressure count.

    The timing includes command recording, submit, GPU completion and CPU
    readback.  It deliberately does not use submit-only FPS.
    """
    # Prime resources and the EventQuery/ring path.
    scene.render_frame_batch(camera_ids)
    pending: list[int] = []
    busy = 0
    submitted = 0
    start = time.perf_counter()
    while submitted < batches:
        token = scene.submit_frame_batch(camera_ids)
        if token:
            pending.append(token)
            submitted += 1
            if len(pending) >= RING_DEPTH:
                scene.read_frame_batch(pending.pop(0))
        else:
            busy += 1
            if pending:
                scene.read_frame_batch(pending.pop(0))
    for token in pending:
        scene.read_frame_batch(token)
    elapsed = time.perf_counter() - start
    camera_fps = len(camera_ids) * batches / elapsed
    return camera_fps, 1000.0 * elapsed / batches, busy


def run_case(num_cameras: int, batches: int) -> dict:
    baseline_mb = median_vram()
    sampler = VramSampler()
    sys.path.insert(0, str(MODULE_DIR))
    import DonutRenderPyNative as rr

    sampler.start()
    try:
        rr.init(runtime_dir=str(REPO_ROOT), backend="vulkan", device_index=-1, enable_debug=False)
        scene = rr.create_scene()
        scene.load_scene(str(SCENE_PATH))
        scene.set_ambient((0.03, 0.04, 0.06), (0.01, 0.01, 0.01))
        scene.set_default_light((-0.4, -1.0, -0.6), (1.0, 1.0, 1.0), 2.0)
        scene.enable_rt_shadows(True)
        scene.enable_shadow_blur(True)
        scene.set_shadow_samples(8)
        scene.set_readback_ring_depth(RING_DEPTH)

        # Exact clones eliminate visibility/content variation between N cases.
        camera_ids = [scene.add_camera(**CAMERA) for _ in range(num_cameras)]
        scene.render_frame_batch(camera_ids)  # loading/AS + render-target warmup
        time.sleep(0.8)                        # ensure steady VRAM samples exist
        resident_mb = median_vram()
        camera_fps, batch_ms, busy = async_e2e_benchmark(scene, camera_ids, batches)
        time.sleep(0.3)
        peak_mb = max(sampler.values, default=resident_mb)
        return {
            "num_cameras": num_cameras,
            "camera_fps": camera_fps,
            "batch_ms": batch_ms,
            "ring_backpressure": busy,
            "baseline_vram_mb": baseline_mb,
            "resident_vram_mb": resident_mb,
            "peak_vram_mb": peak_mb,
            "resident_delta_mb": max(0, resident_mb - baseline_mb),
            "peak_delta_mb": max(0, peak_mb - baseline_mb),
            "vram_samples": sampler.values,
        }
    finally:
        try:
            rr.destroy()
        finally:
            sampler.stop()


def make_chart(results: list[dict]) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ns = [r["num_cameras"] for r in results]
    fps = [r["camera_fps"] for r in results]
    fps_std = [r.get("camera_fps_std", 0.0) for r in results]
    vram_delta = [r["resident_delta_mb"] for r in results]
    peak_delta = [r["peak_delta_mb"] for r in results]
    resident_growth = [value - vram_delta[0] for value in vram_delta]
    peak_growth = [value - peak_delta[0] for value in peak_delta]
    baseline = fps[0]
    ideal = [baseline * n for n in ns]
    speedup = [value / baseline for value in fps]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(ns))
    width = 0.35
    ax1.bar(x - width / 2, fps, width, yerr=fps_std, capsize=3,
            label="Async end-to-end throughput", color="#4a90d9")
    ax1.bar(x + width / 2, ideal, width, label="Ideal (linear)", color="#d9d9d9", hatch="//", edgecolor="#888")
    ax1.set_xticks(x, [f"N={n}" for n in ns])
    ax1.set_ylabel("Camera Throughput (FPS)")
    ax1.set_title("ReplicaCAD Async Batch Throughput\n(RT shadow 8-sample, no GI, .60 HBD, end-to-end)")
    ax1.legend(loc="upper left")
    ax1.grid(axis="y", alpha=0.3)
    for i, (value, ratio) in enumerate(zip(fps, speedup)):
        ax1.text(i - width / 2, value + max(fps) * 0.015, f"{value:.1f}\n({ratio:.2f}x)",
                 ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax2.bar(ns, resident_growth, color="#27ae60", alpha=0.85,
            label="Resident VRAM growth")
    ax2.scatter(ns, peak_growth, color="#e74c3c", zorder=3,
                label="Peak VRAM growth")
    ax2.set_xlabel("Number of camera slots")
    ax2.set_ylabel("VRAM growth from N=1 (MB)")
    ax2.set_title("Shared-scene GPU Memory Growth\n(ReplicaCAD scene/BLAS/TLAS loaded once)")
    ax2.set_xticks(ns)
    ax2.set_ylim(0, max(30, max(peak_growth) * 1.22))
    ax2.grid(axis="y", alpha=0.3)
    ax2.legend(loc="upper left")
    for n, resident, peak in zip(ns, resident_growth, peak_growth):
        ax2.text(n, max(resident, peak) + max(12, max(peak_growth) * 0.025),
                 f"+{resident} / +{peak} MB",
                 ha="center", fontsize=8)
    ax2.text(0.02, 0.02,
             f"median of 3 trials; N=1: {vram_delta[0]}/{peak_delta[0]} MB vs idle, "
             f"N={ns[-1]}: {vram_delta[-1]}/{peak_delta[-1]} MB vs idle\n"
             "labels: resident / peak growth relative to N=1",
             transform=ax2.transAxes,
             fontsize=8, color="#555")

    fig.suptitle("Flora ReplicaCAD Parallel Rendering Scaling", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = OUT_DIR / "replicacad_parallel_throughput_vram.png"
    fig.savefig(path, dpi=140)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cameras", type=int, nargs="+", default=[1, 2, 4, 6, 8])
    parser.add_argument("--batches", type=int, default=60)
    parser.add_argument("--trials", type=int, default=3)
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not SCENE_PATH.exists():
        raise FileNotFoundError(SCENE_PATH)

    print("=== Flora ReplicaCAD async-batch scaling ===")
    print(f"scene={SCENE_PATH.name}, {WIDTH}x{HEIGHT}, RT shadow=8, GI=off, HBD exposure=.60")
    print(f"camera slots={args.cameras}, batches/case={args.batches}, trials={args.trials}, ring K={RING_DEPTH}")
    results = []
    for n in args.cameras:
        print(f"\n--- N={n} ---", flush=True)
        trials = []
        for trial in range(args.trials):
            result = run_case(n, args.batches)
            trials.append(result)
            print(f"trial {trial + 1}/{args.trials}: throughput={result['camera_fps']:.1f} cam-FPS, "
                  f"batch={result['batch_ms']:.2f}ms, VRAM resident/peak "
                  f"Δ={result['resident_delta_mb']}/{result['peak_delta_mb']}MB, "
                  f"ring busy={result['ring_backpressure']}", flush=True)
        aggregate = {
            "num_cameras": n,
            "camera_fps": float(np.mean([r["camera_fps"] for r in trials])),
            "camera_fps_std": float(np.std([r["camera_fps"] for r in trials], ddof=0)),
            "batch_ms": float(np.mean([r["batch_ms"] for r in trials])),
            "ring_backpressure": int(sum(r["ring_backpressure"] for r in trials)),
            "baseline_vram_mb": int(round(statistics.median(r["baseline_vram_mb"] for r in trials))),
            "resident_vram_mb": int(round(statistics.median(r["resident_vram_mb"] for r in trials))),
            "peak_vram_mb": int(round(statistics.median(r["peak_vram_mb"] for r in trials))),
            "resident_delta_mb": int(round(statistics.median(r["resident_delta_mb"] for r in trials))),
            "peak_delta_mb": int(round(statistics.median(r["peak_delta_mb"] for r in trials))),
            "trials": trials,
        }
        results.append(aggregate)
        print(f"mean: throughput={aggregate['camera_fps']:.1f}±{aggregate['camera_fps_std']:.1f} cam-FPS, "
              f"VRAM resident/peak Δ={aggregate['resident_delta_mb']}/{aggregate['peak_delta_mb']}MB", flush=True)

    base = results[0]["camera_fps"]
    for result in results:
        result["speedup"] = result["camera_fps"] / base
    json_path = OUT_DIR / "replicacad_parallel_scaling_results.json"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    chart_path = make_chart(results)
    print(f"\nSaved: {json_path}")
    print(f"Saved: {chart_path}")


if __name__ == "__main__":
    main()
