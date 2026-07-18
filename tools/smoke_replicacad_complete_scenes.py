"""Compile, load and render every complete ReplicaCAD scene, including URDF visuals."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = REPO_ROOT / "ReplicaCAD"
DEFAULT_OUTPUT = REPO_ROOT / "output" / "replicacad_a3" / "all_scene_smoke.json"
MODULE_DIR = REPO_ROOT / "bin" / "windows-x64"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.width <= 0 or args.height <= 0:
        raise ValueError("width and height must be positive.")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("limit must be positive when provided.")

    sys.path.insert(0, str(REPO_ROOT / "python"))
    sys.path.insert(0, str(MODULE_DIR))
    from donut_render_py import ReplicaCADManifest, compile_donut_scene
    import DonutRenderPyNative as rr

    manifest = ReplicaCADManifest.from_dataset_root(args.dataset)
    handles = manifest.scene_handles
    if args.limit is not None:
        handles = handles[: args.limit]
    total = len(handles)
    results = []
    started = time.perf_counter()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = REPO_ROOT / "output"
    temporary_root.mkdir(parents=True, exist_ok=True)
    rr.init(
        runtime_dir=str(REPO_ROOT),
        backend="vulkan",
        device_index=-1,
        enable_debug=False,
    )
    native_scene = None
    try:
        native_scene = rr.create_scene()
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
        native_scene.set_ambient((0.03, 0.04, 0.06), (0.01, 0.01, 0.01))
        native_scene.set_default_light((-0.4, -1.0, -0.6), (1.0, 1.0, 1.0), 2.0)
        native_scene.enable_rt_shadows(False)

        with tempfile.TemporaryDirectory(dir=temporary_root) as directory:
            scene_path = Path(directory) / "current.donut_scene.json"
            for index, handle in enumerate(handles, start=1):
                scene_start = time.perf_counter()
                scene_desc = manifest.parse_scene(handle)
                compiled = compile_donut_scene(scene_desc)
                artifact = compiled.write(scene_path)
                load_start = time.perf_counter()
                native_scene.load_scene(str(artifact.scene_path))
                load_ms = 1000.0 * (time.perf_counter() - load_start)
                stats = native_scene.get_scene_stats()
                control_handles = native_scene.get_node_handles(compiled.control_node_names)
                if len(control_handles) != len(set(control_handles)):
                    raise RuntimeError(f"Scene {handle} has duplicate native control handles.")
                sample = compiled.instances[min(1, len(compiled.instances) - 1)]
                native_matrix = native_scene.get_node_world_transform(sample.node_name)
                transform_error = max(
                    abs(actual - expected)
                    for actual, expected in zip(
                        native_matrix,
                        (value for row in sample.asset_matrix_row_major for value in row),
                    )
                )
                render_start = time.perf_counter()
                pixels = native_scene.render_frame()
                render_ms = 1000.0 * (time.perf_counter() - render_start)
                if len(pixels) != args.width * args.height * 4:
                    raise RuntimeError(
                        f"Unexpected readback size for {handle}: {len(pixels)} bytes."
                    )
                if stats["mesh_instances"] <= 0:
                    raise RuntimeError(f"Scene {handle} produced no mesh instances.")
                if transform_error >= 1.0e-5:
                    raise RuntimeError(
                        f"Scene {handle} transform error {transform_error:.3e} exceeds 1e-5."
                    )
                total_ms = 1000.0 * (time.perf_counter() - scene_start)
                results.append(
                    {
                        "scene": handle,
                        "ordinary_objects": len(scene_desc.objects),
                        "articulated_instances": compiled.articulated_instance_count,
                        "articulated_links": compiled.articulated_link_count,
                        "articulated_visuals": compiled.articulated_visual_count,
                        "omitted_articulated_instances": compiled.omitted_articulated_instances,
                        "unique_models": compiled.model_count,
                        "render_instances": compiled.render_instance_count,
                        "control_nodes": len(control_handles),
                        "native_node_handles": native_scene.node_handle_count,
                        "native_mesh_instances": stats["mesh_instances"],
                        "load_ms": load_ms,
                        "render_ms": render_ms,
                        "transform_error": transform_error,
                        "total_ms": total_ms,
                    }
                )
                print(
                    f"[{index:02d}/{total:02d}] {handle}: "
                    f"models={compiled.model_count}, instances={compiled.render_instance_count}, "
                    f"articulated={compiled.articulated_instance_count}, "
                    f"load={load_ms:.1f} ms, render={render_ms:.1f} ms",
                    flush=True,
                )
    finally:
        native_scene = None
        rr.destroy()

    elapsed_ms = 1000.0 * (time.perf_counter() - started)
    report = {
        "schema_version": 2,
        "resolution": [args.width, args.height],
        "summary": {
            "requested_scenes": total,
            "passed_scenes": len(results),
            "ordinary_objects": sum(item["ordinary_objects"] for item in results),
            "articulated_instances": sum(
                item["articulated_instances"] for item in results
            ),
            "articulated_links": sum(item["articulated_links"] for item in results),
            "articulated_visuals": sum(
                item["articulated_visuals"] for item in results
            ),
            "omitted_articulated_instances": sum(
                item["omitted_articulated_instances"] for item in results
            ),
            "max_transform_error": max(
                (item["transform_error"] for item in results), default=0.0
            ),
            "mean_load_ms": (
                sum(item["load_ms"] for item in results) / len(results)
                if results
                else 0.0
            ),
            "max_load_ms": max((item["load_ms"] for item in results), default=0.0),
            "elapsed_ms": elapsed_ms,
        },
        "scenes": results,
    }
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"PASS: {len(results)}/{total} scenes loaded and rendered in "
        f"{elapsed_ms / 1000.0:.2f} s. Report: {args.output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
