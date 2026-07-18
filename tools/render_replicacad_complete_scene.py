"""Compile and render one ReplicaCAD static scene through Donut's model instancing."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = REPO_ROOT / "ReplicaCAD"
DEFAULT_OUTPUT = REPO_ROOT / "output" / "replicacad_complete"
MODULE_DIR = REPO_ROOT / "bin" / "windows-x64"


def repo_path(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--scene", default="apt_0")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--rt-shadows", action="store_true")
    parser.add_argument("--compile-only", action="store_true")
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


def main() -> int:
    args = parse_args()
    if args.width <= 0 or args.height <= 0 or args.frames <= 0 or args.warmup < 0:
        raise ValueError("width, height and frames must be positive; warmup must be non-negative.")

    sys.path.insert(0, str(REPO_ROOT / "python"))
    from donut_render_py import ReplicaCADManifest, compile_donut_scene

    print(f"[1/5] Parsing ReplicaCAD scene {args.scene!r}...", flush=True)
    compile_start = time.perf_counter()
    manifest = ReplicaCADManifest.from_dataset_root(args.dataset)
    scene_desc = manifest.parse_scene(args.scene)
    compiled = compile_donut_scene(scene_desc)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scene_path = args.output_dir / f"{scene_desc.name}.donut_scene.json"
    artifact = compiled.write(scene_path)
    compile_ms = 1000.0 * (time.perf_counter() - compile_start)
    print(
        f"[2/5] Compiled {compiled.instance_count} render instances from "
        f"{compiled.model_count} unique GLBs in {compile_ms:.2f} ms.",
        flush=True,
    )
    print(f"      Scene: {artifact.scene_path}", flush=True)

    if args.compile_only:
        print("PASS: compile-only requested; native rendering skipped.", flush=True)
        return 0

    print("[3/5] Initializing native Vulkan renderer...", flush=True)
    sys.path.insert(0, str(MODULE_DIR))
    import DonutRenderPyNative as rr

    vram_before = query_vram_mb()
    rr.init(
        runtime_dir=str(REPO_ROOT),
        backend="vulkan",
        device_index=-1,
        enable_debug=False,
    )
    native_scene = None
    try:
        native_scene = rr.create_scene()
        print("[4/5] Loading unique GLBs and instancing SceneGraph...", flush=True)
        load_start = time.perf_counter()
        native_scene.load_scene(str(artifact.scene_path))
        load_ms = 1000.0 * (time.perf_counter() - load_start)
        vram_loaded = query_vram_mb()
        scene_stats = native_scene.get_scene_stats()
        transform_errors = []
        for instance in compiled.instances[:20]:
            native_matrix = np.asarray(
                native_scene.get_node_world_transform(instance.node_name),
                dtype=np.float64,
            ).reshape(4, 4)
            reference_matrix = np.asarray(
                instance.asset_matrix_row_major, dtype=np.float64
            )
            transform_errors.append(float(np.max(np.abs(native_matrix - reference_matrix))))
        max_transform_error = max(transform_errors, default=0.0)
        if max_transform_error >= 1.0e-5:
            raise RuntimeError(
                f"Native SceneGraph transform error {max_transform_error:.3e} exceeds 1e-5."
            )
        print(
            f"      Native load completed in {load_ms:.2f} ms; "
            f"mesh instances={scene_stats['mesh_instances']}, "
            f"transform max error={max_transform_error:.3e}.",
            flush=True,
        )

        native_scene.set_camera(
            position=(3.5, 2.0, 3.5),
            target=(0.0, 1.0, 0.0),
            up=(0.0, 1.0, 0.0),
            fov_degrees=60.0,
            width=args.width,
            height=args.height,
            z_near=0.1,
            z_far=100.0,
        )
        # Match the established Flora/SAPIEN ReplicaCAD comparison setup.
        native_scene.set_ambient((0.03, 0.04, 0.06), (0.01, 0.01, 0.01))
        native_scene.set_default_light((-0.4, -1.0, -0.6), (1.0, 1.0, 1.0), 2.0)
        native_scene.enable_rt_shadows(args.rt_shadows)
        native_scene.enable_shadow_blur(args.rt_shadows)
        if args.rt_shadows:
            native_scene.set_shadow_samples(8)

        print("[5/5] Rendering and reading back frames...", flush=True)
        first_frame_start = time.perf_counter()
        image_bytes = native_scene.render_frame()
        first_frame_ms = 1000.0 * (time.perf_counter() - first_frame_start)
        for _ in range(args.warmup):
            native_scene.render_frame()
        frame_times = []
        for _ in range(args.frames):
            start = time.perf_counter()
            image_bytes = native_scene.render_frame()
            frame_times.append(1000.0 * (time.perf_counter() - start))

        rgba = np.frombuffer(image_bytes, dtype=np.uint8).reshape(
            args.height, args.width, 4
        )
        rgb = np.ascontiguousarray(rgba[:, :, :3])
        variant = "rt" if args.rt_shadows else "raster"
        image_path = args.output_dir / f"{scene_desc.name}_complete_{variant}.png"
        Image.fromarray(rgb, "RGB").save(image_path)
        metrics = {
            "scene": scene_desc.name,
            "resolution": [args.width, args.height],
            "rt_shadows": bool(args.rt_shadows),
            "camera": {
                "position": [3.5, 2.0, 3.5],
                "target": [0.0, 1.0, 0.0],
                "vertical_fov_degrees": 60.0,
            },
            "lighting": {
                "ambient_top": [0.03, 0.04, 0.06],
                "ambient_bottom": [0.01, 0.01, 0.01],
                "direction": [-0.4, -1.0, -0.6],
                "irradiance": 2.0,
            },
            "compile_ms": compile_ms,
            "load_ms": load_ms,
            "first_frame_ms": first_frame_ms,
            "mean_frame_ms": float(np.mean(frame_times)),
            "camera_fps": 1000.0 / float(np.mean(frame_times)),
            "unique_models": compiled.model_count,
            "render_instances": compiled.instance_count,
            "ordinary_objects": len(scene_desc.objects),
            "omitted_articulated_instances": compiled.omitted_articulated_instances,
            "native_scene_stats": scene_stats,
            "validated_transforms": len(transform_errors),
            "max_transform_error": max_transform_error,
            "vram_before_mb": vram_before,
            "vram_loaded_mb": vram_loaded,
            "vram_load_delta_mb": (
                None
                if vram_before is None or vram_loaded is None
                else max(0, vram_loaded - vram_before)
            ),
            "rgb_mean": float(rgb.mean()),
            "rgb_std": float(rgb.std()),
            "nonblack_fraction": float((rgb.sum(axis=2) > 10).mean()),
            "scene_path": repo_path(artifact.scene_path),
            "image_path": repo_path(image_path),
            "determinism_digest": artifact.determinism_digest,
        }
        metrics_path = args.output_dir / f"{scene_desc.name}_{variant}_metrics.json"
        metrics_path.write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(
            f"PASS: image={image_path}, mean={metrics['rgb_mean']:.2f}, "
            f"nonblack={100.0 * metrics['nonblack_fraction']:.1f}%, "
            f"mean frame={metrics['mean_frame_ms']:.2f} ms",
            flush=True,
        )
        print(f"Metrics: {metrics_path}", flush=True)
        return 0
    finally:
        native_scene = None
        rr.destroy()


if __name__ == "__main__":
    raise SystemExit(main())
