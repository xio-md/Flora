"""Render closed/open ReplicaCAD articulation states through Flora's batch pose API."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = REPO_ROOT / "ReplicaCAD"
MODULE_DIR = REPO_ROOT / "bin" / "windows-x64"
OUTPUT_DIR = REPO_ROOT / "output" / "replicacad_a3"


OPEN_JOINTS: dict[str, dict[str, float]] = {
    "fridge": {"top_door_hinge": 1.15},
    "kitchen_counter": {"left_slide_top": 0.38},
    "kitchenCupboard_01": {
        "doorWhole_1L_hinge": -1.05,
        "doorWhole_1R_hinge": 1.05,
    },
    "chestOfDrawers_01": {"drawer_topL": 0.38},
    "cabinet": {"left_slide": 0.52},
    "door2": {"root_rotation": -1.0},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET_ROOT)
    parser.add_argument("--scene", default="apt_0")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--rt-shadows", action="store_true")
    return parser.parse_args()


def flatten_matrix(matrix: tuple[tuple[float, ...], ...]) -> list[float]:
    return [float(value) for row in matrix for value in row]


def frame_array(frame: bytes, width: int, height: int) -> np.ndarray:
    return np.frombuffer(frame, dtype=np.uint8).reshape(height, width, 4).copy()


def portable(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def main() -> int:
    args = parse_args()
    if args.width <= 0 or args.height <= 0:
        raise ValueError("Resolution must be positive.")

    sys.path.insert(0, str(REPO_ROOT / "python"))
    sys.path.insert(0, str(MODULE_DIR))
    from donut_render_py import ReplicaCADManifest, compile_donut_scene
    import DonutRenderPyNative as rr

    manifest = ReplicaCADManifest.from_dataset_root(args.dataset)
    compiled = compile_donut_scene(manifest.parse_scene(args.scene))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifact = compiled.write(args.output_dir / f"{args.scene}.donut_scene.json")

    positions: dict[int, dict[str, float]] = {}
    updated_joint_count = 0
    for articulation in compiled.articulations:
        joint_positions = OPEN_JOINTS.get(articulation.template_name)
        if joint_positions:
            positions[articulation.instance_id] = joint_positions
            updated_joint_count += len(joint_positions)
    update_names, update_matrices = compiled.joint_transform_updates(positions)

    rr.init(
        runtime_dir=str(REPO_ROOT),
        backend="vulkan",
        device_index=-1,
        enable_debug=False,
    )
    native_scene = None
    try:
        native_scene = rr.create_scene()
        load_start = time.perf_counter()
        native_scene.load_scene(str(artifact.scene_path))
        load_ms = 1000.0 * (time.perf_counter() - load_start)
        native_scene.set_camera(
            position=(3.5, 2.0, 3.5),
            target=(0.0, 1.0, 0.0),
            up=(0.0, 1.0, 0.0),
            fov_degrees=60.0,
            width=args.width,
            height=args.height,
            z_near=0.05,
            z_far=50.0,
        )
        native_scene.set_ambient(
            top_rgb=(0.03, 0.04, 0.06),
            bottom_rgb=(0.01, 0.01, 0.01),
        )
        native_scene.set_default_light(
            direction=(-0.4, -1.0, -0.6),
            color=(1.0, 1.0, 1.0),
            irradiance=2.0,
        )
        native_scene.enable_rt_shadows(args.rt_shadows)
        native_scene.set_shadow_samples(8)

        resolve_start = time.perf_counter()
        control_handles = native_scene.get_node_handles(compiled.control_node_names)
        resolve_ms = 1000.0 * (time.perf_counter() - resolve_start)
        if len(set(control_handles)) != len(control_handles):
            raise RuntimeError("Compiled control nodes did not resolve to unique native handles.")
        handle_by_name = dict(zip(compiled.control_node_names, control_handles))
        update_handles = [handle_by_name[name] for name in update_names]

        closed = frame_array(
            native_scene.render_frame(), args.width, args.height
        )
        before_world = {
            name: native_scene.get_node_world_transform_by_handle(handle)
            for name, handle in zip(update_names, update_handles)
        }

        matrix_values = [flatten_matrix(matrix) for matrix in update_matrices]
        update_start = time.perf_counter()
        native_scene.update_node_transforms_batch(update_handles, matrix_values)
        update_ms = 1000.0 * (time.perf_counter() - update_start)

        opened = frame_array(
            native_scene.render_frame(), args.width, args.height
        )
        after_world = {
            name: native_scene.get_node_world_transform_by_handle(handle)
            for name, handle in zip(update_names, update_handles)
        }

        world_deltas = {
            name: float(
                np.max(
                    np.abs(
                        np.asarray(after_world[name], dtype=np.float64)
                        - np.asarray(before_world[name], dtype=np.float64)
                    )
                )
            )
            for name in update_names
        }
        if not world_deltas or max(world_deltas.values()) <= 1.0e-5:
            raise RuntimeError("Joint updates did not change native world transforms.")

        suffix = "rt" if args.rt_shadows else "raster"
        closed_path = args.output_dir / f"{args.scene}_articulation_closed_{suffix}.png"
        open_path = args.output_dir / f"{args.scene}_articulation_open_{suffix}.png"
        Image.fromarray(closed[:, :, :3], "RGB").save(closed_path)
        Image.fromarray(opened[:, :, :3], "RGB").save(open_path)

        rgb_delta = np.abs(
            opened[:, :, :3].astype(np.int16) - closed[:, :, :3].astype(np.int16)
        )
        metrics = {
            "schema_version": 1,
            "scene": compiled.scene_name,
            "mode": suffix,
            "resolution": [args.width, args.height],
            "scene_path": portable(artifact.scene_path),
            "closed_image": portable(closed_path),
            "open_image": portable(open_path),
            "load_ms": load_ms,
            "unique_models": compiled.model_count,
            "render_instances": compiled.render_instance_count,
            "graph_nodes": compiled.graph_node_count,
            "articulated_instances": compiled.articulated_instance_count,
            "articulated_links": compiled.articulated_link_count,
            "articulated_visuals": compiled.articulated_visual_count,
            "compiled_control_nodes": len(compiled.control_node_names),
            "native_node_handles": native_scene.node_handle_count,
            "handle_resolve_ms": resolve_ms,
            "updated_joints": updated_joint_count,
            "pose_update_call_ms": update_ms,
            "max_world_transform_delta": max(world_deltas.values()),
            "mean_rgb_delta": float(rgb_delta.mean()),
            "changed_pixel_ratio": float(np.any(rgb_delta > 5, axis=2).mean()),
            "native_scene_stats": native_scene.get_scene_stats(),
        }
        metrics_path = args.output_dir / f"{args.scene}_articulation_{suffix}_metrics.json"
        metrics_path.write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(
            "PASS: "
            f"articulations={compiled.articulated_instance_count}, "
            f"links={compiled.articulated_link_count}, "
            f"visuals={compiled.articulated_visual_count}, "
            f"updated_joints={updated_joint_count}, "
            f"pose_update={update_ms:.3f} ms, "
            f"changed_pixels={100.0 * metrics['changed_pixel_ratio']:.2f}%",
            flush=True,
        )
        print(f"Closed: {closed_path}", flush=True)
        print(f"Open:   {open_path}", flush=True)
        print(f"Metrics:{metrics_path}", flush=True)
        return 0
    finally:
        native_scene = None
        rr.destroy()


if __name__ == "__main__":
    raise SystemExit(main())
