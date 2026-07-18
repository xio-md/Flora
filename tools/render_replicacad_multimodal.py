"""Render and validate aligned ReplicaCAD multimodal sensor products."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = REPO_ROOT / "ReplicaCAD"
MODULE_DIR = REPO_ROOT / "bin" / "windows-x64"
OUTPUT_DIR = REPO_ROOT / "output" / "replicacad_a4"
PRODUCTS = ("color", "depth", "normal", "instance", "semantic")

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
    parser.add_argument("--depth-visual-max", type=float, default=10.0)
    parser.add_argument("--rt-shadows", action="store_true")
    return parser.parse_args()


def flatten_matrix(matrix: tuple[tuple[float, ...], ...]) -> list[float]:
    return [float(value) for row in matrix for value in row]


def portable(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def id_visual(ids: np.ndarray) -> np.ndarray:
    values = ids.astype(np.uint64)
    rgb = np.stack(
        (
            (values * 37 + 17) % 251,
            (values * 67 + 43) % 253,
            (values * 97 + 71) % 255,
        ),
        axis=2,
    ).astype(np.uint8)
    rgb[ids == 0] = 0
    return rgb


def depth_visual(depth: np.ndarray, depth_max: float) -> np.ndarray:
    valid = depth > 0.0
    intensity = np.zeros(depth.shape, dtype=np.uint8)
    intensity[valid] = np.clip(
        255.0 * (1.0 - depth[valid] / depth_max), 0.0, 255.0
    ).astype(np.uint8)
    return np.repeat(intensity[:, :, None], 3, axis=2)


def normal_visual(normal: np.ndarray) -> np.ndarray:
    lengths = np.linalg.norm(normal, axis=2)
    rgb = np.clip((normal * 0.5 + 0.5) * 255.0, 0.0, 255.0).astype(np.uint8)
    rgb[lengths <= 0.0] = 0
    return rgb


def frame_images(frame, depth_max: float) -> dict[str, np.ndarray]:
    return {
        "color": frame.color[:, :, :3],
        "depth": depth_visual(frame.depth, depth_max),
        "normal": normal_visual(frame.normal),
        "instance": id_visual(frame.instance),
        "semantic": id_visual(frame.semantic),
    }


def validate_frame(frame, labels) -> tuple[dict[str, object], list[dict[str, object]]]:
    expected_shapes = {
        "color": (frame.height, frame.width, 4),
        "depth": (frame.height, frame.width),
        "normal": (frame.height, frame.width, 3),
        "instance": (frame.height, frame.width),
        "semantic": (frame.height, frame.width),
    }
    expected_dtypes = {
        "color": np.dtype(np.uint8),
        "depth": np.dtype(np.float32),
        "normal": np.dtype(np.float32),
        "instance": np.dtype(np.uint32),
        "semantic": np.dtype(np.uint32),
    }
    for product in PRODUCTS:
        array = getattr(frame, product)
        if array is None:
            raise RuntimeError(f"Missing sensor product: {product}")
        if array.shape != expected_shapes[product]:
            raise RuntimeError(
                f"{product} has shape {array.shape}; expected {expected_shapes[product]}."
            )
        if array.dtype != expected_dtypes[product]:
            raise RuntimeError(
                f"{product} has dtype {array.dtype}; expected {expected_dtypes[product]}."
            )
        if not array.flags.c_contiguous or not array.flags.owndata:
            raise RuntimeError(f"{product} must be contiguous and own its storage.")

    depth_valid = frame.depth > 0.0
    normal_lengths = np.linalg.norm(frame.normal, axis=2)
    normal_valid = normal_lengths > 0.5
    instance_valid = frame.instance > 0
    if not np.array_equal(depth_valid, normal_valid):
        raise RuntimeError("Depth and normal valid-pixel masks are not aligned.")
    if not np.array_equal(depth_valid, instance_valid):
        raise RuntimeError("Depth and instance valid-pixel masks are not aligned.")
    if not depth_valid.any():
        raise RuntimeError("Sensor frame contains no visible geometry.")
    if np.any(frame.depth[~depth_valid] != 0.0):
        raise RuntimeError("Depth background must be 0.")
    if np.any(frame.normal[~depth_valid] != 0.0):
        raise RuntimeError("Normal background must be (0, 0, 0).")
    if np.any(frame.semantic[~depth_valid] != 0):
        raise RuntimeError("Semantic background must be 0.")

    max_label_id = max(label.instance_id for label in labels)
    semantic_lut = np.zeros(max_label_id + 1, dtype=np.uint32)
    label_by_id = {}
    for label in labels:
        semantic_lut[label.instance_id] = label.semantic_id
        label_by_id[label.instance_id] = label
    if int(frame.instance.max()) > max_label_id:
        raise RuntimeError("Native instance output contains an unregistered ID.")
    expected_semantic = semantic_lut[frame.instance]
    if not np.array_equal(frame.semantic, expected_semantic):
        mismatch = int(np.count_nonzero(frame.semantic != expected_semantic))
        raise RuntimeError(f"Instance-to-semantic mapping mismatches at {mismatch} pixels.")

    visible_ids = [int(value) for value in np.unique(frame.instance) if value != 0]
    if len(visible_ids) < 20:
        raise RuntimeError(
            f"Only {len(visible_ids)} labeled objects are visible; expected at least 20."
        )
    samples: list[dict[str, object]] = []
    for instance_id in visible_ids[:20]:
        ys, xs = np.nonzero(frame.instance == instance_id)
        center = np.asarray([ys.mean(), xs.mean()])
        index = int(
            np.argmin((ys - center[0]) ** 2 + (xs - center[1]) ** 2)
        )
        y, x = int(ys[index]), int(xs[index])
        label = label_by_id[instance_id]
        samples.append(
            {
                "instance_id": instance_id,
                "semantic_id": int(frame.semantic[y, x]),
                "node_name": label.node_name,
                "kind": label.kind,
                "representative_pixel_xy": [x, y],
            }
        )

    unit_errors = np.abs(normal_lengths[normal_valid] - 1.0)
    metrics = {
        "resolution": [frame.width, frame.height],
        "products": {
            product: {
                "shape": list(getattr(frame, product).shape),
                "dtype": str(getattr(frame, product).dtype),
            }
            for product in PRODUCTS
        },
        "valid_pixels": int(depth_valid.sum()),
        "valid_pixel_ratio": float(depth_valid.mean()),
        "visible_instance_count": len(visible_ids),
        "visible_semantic_count": int(np.unique(frame.semantic[depth_valid]).size),
        "depth_min_m": float(frame.depth[depth_valid].min()),
        "depth_max_m": float(frame.depth[depth_valid].max()),
        "normal_unit_error_mean": float(unit_errors.mean()),
        "normal_unit_error_max": float(unit_errors.max()),
        "aligned_valid_masks": True,
        "background_zero": True,
        "instance_semantic_mapping_exact": True,
    }
    return metrics, samples


def make_contact_sheet(
    states: dict[str, dict[str, np.ndarray]], output_path: Path
) -> None:
    tile_width = 320
    first_image = next(iter(next(iter(states.values())).values()))
    tile_height = max(1, round(tile_width * first_image.shape[0] / first_image.shape[1]))
    header_height = 28
    sheet = Image.new(
        "RGB",
        (tile_width * len(PRODUCTS), (tile_height + header_height) * len(states)),
        (245, 245, 245),
    )
    draw = ImageDraw.Draw(sheet)
    for row, (state, images) in enumerate(states.items()):
        y = row * (tile_height + header_height)
        for column, product in enumerate(PRODUCTS):
            x = column * tile_width
            draw.text((x + 7, y + 7), f"{state}: {product}", fill=(25, 25, 25))
            tile = Image.fromarray(images[product], "RGB").resize(
                (tile_width, tile_height), Image.Resampling.LANCZOS
            )
            sheet.paste(tile, (x, y + header_height))
    sheet.save(output_path)


def main() -> int:
    args = parse_args()
    if args.width <= 0 or args.height <= 0 or args.depth_visual_max <= 0.0:
        raise ValueError("Resolution and depth visualization range must be positive.")

    sys.path.insert(0, str(REPO_ROOT / "python"))
    sys.path.insert(0, str(MODULE_DIR))
    from donut_render_py import ReplicaCADManifest, compile_donut_scene
    from rtxns_genesis_style.sensor import decode_sensor_frame
    import DonutRenderPyNative as rr

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = ReplicaCADManifest.from_dataset_root(args.dataset)
    compiled = compile_donut_scene(manifest.parse_scene(args.scene))
    artifact = compiled.write(args.output_dir / f"{args.scene}.donut_scene.json")

    open_positions: dict[int, dict[str, float]] = {}
    for articulation in compiled.articulations:
        positions = OPEN_JOINTS.get(articulation.template_name)
        if positions:
            open_positions[articulation.instance_id] = positions
    update_names, update_matrices = compiled.joint_transform_updates(open_positions)

    rr.init(
        runtime_dir=str(REPO_ROOT),
        backend="vulkan",
        device_index=-1,
        enable_debug=False,
    )
    scene = None
    try:
        scene = rr.create_scene()
        load_started = time.perf_counter()
        scene.load_scene(str(artifact.scene_path))
        load_ms = 1000.0 * (time.perf_counter() - load_started)
        first_two_labels = compiled.sensor_labels[:2]
        try:
            scene.set_node_labels(
                [label.node_name for label in first_two_labels],
                [1, 1],
                [label.semantic_id for label in first_two_labels],
            )
        except ValueError:
            duplicate_id_rejected = True
        else:
            duplicate_id_rejected = False
        compiled.configure_sensor_labels(scene)
        scene.set_camera(
            position=(3.5, 2.0, 3.5),
            target=(0.0, 1.0, 0.0),
            up=(0.0, 1.0, 0.0),
            fov_degrees=60.0,
            width=args.width,
            height=args.height,
            z_near=0.05,
            z_far=50.0,
        )
        scene.set_ambient((0.03, 0.04, 0.06), (0.01, 0.01, 0.01))
        scene.set_default_light((-0.4, -1.0, -0.6), (1.0, 1.0, 1.0), 2.0)
        scene.enable_rt_shadows(args.rt_shadows)
        scene.enable_shadow_blur(True)
        scene.set_shadow_samples(8)
        handles = scene.get_node_handles(update_names)

        raw_closed = scene.render_sensor_batch([0], list(PRODUCTS))[0]
        closed = decode_sensor_frame(raw_closed)
        closed_stats = scene.get_last_frame_stats()

        scene.update_node_transforms_batch(
            handles, [flatten_matrix(matrix) for matrix in update_matrices]
        )
        raw_open = scene.render_sensor_batch([0], list(PRODUCTS))[0]
        opened = decode_sensor_frame(raw_open)
        open_stats = scene.get_last_frame_stats()

        closed_metrics, closed_samples = validate_frame(closed, compiled.sensor_labels)
        open_metrics, open_samples = validate_frame(opened, compiled.sensor_labels)
        state_frames = {"closed": closed, "open": opened}
        state_images = {
            name: frame_images(frame, args.depth_visual_max)
            for name, frame in state_frames.items()
        }
        image_paths: dict[str, dict[str, str]] = {}
        for state, images in state_images.items():
            image_paths[state] = {}
            for product, image in images.items():
                path = args.output_dir / f"{args.scene}_{state}_{product}.png"
                Image.fromarray(image, "RGB").save(path)
                image_paths[state][product] = portable(path)
        contact_path = args.output_dir / f"{args.scene}_multimodal_contact_sheet.png"
        make_contact_sheet(state_images, contact_path)

        dynamic_delta = {
            product: float(
                np.any(
                    getattr(closed, product) != getattr(opened, product),
                    axis=2,
                ).mean()
                if getattr(closed, product).ndim == 3
                else (getattr(closed, product) != getattr(opened, product)).mean()
            )
            for product in PRODUCTS
        }
        checks = {
            "duplicate_instance_id_rejected": duplicate_id_rejected,
            "closed_products_valid": True,
            "open_products_valid": True,
            "at_least_20_visible_instances": min(
                closed_metrics["visible_instance_count"],
                open_metrics["visible_instance_count"],
            )
            >= 20,
            "dynamic_geometry_changes_geometric_products": all(
                dynamic_delta[product] > 0.0
                for product in ("color", "depth", "normal", "instance")
            ),
            "semantic_mapping_remains_exact": True,
        }
        report = {
            "schema_version": 1,
            "scene": compiled.scene_name,
            "mode": "rt_shadow" if args.rt_shadows else "raster",
            "camera": {
                "position": [3.5, 2.0, 3.5],
                "target": [0.0, 1.0, 0.0],
                "up": [0.0, 1.0, 0.0],
                "fov_degrees": 60.0,
                "z_near": 0.05,
                "z_far": 50.0,
            },
            "load_ms": load_ms,
            "scene_path": portable(artifact.scene_path),
            "sensor_label_count": len(compiled.sensor_labels),
            "updated_joint_count": len(update_names),
            "closed": closed_metrics,
            "open": open_metrics,
            "closed_representative_samples": closed_samples,
            "open_representative_samples": open_samples,
            "dynamic_changed_pixel_ratio": dynamic_delta,
            "native_frame_stats": {"closed": closed_stats, "open": open_stats},
            "images": image_paths,
            "contact_sheet": portable(contact_path),
            "checks": checks,
            "passed": all(checks.values()),
        }
        metrics_path = args.output_dir / f"{args.scene}_multimodal_metrics.json"
        metrics_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        if not report["passed"]:
            failed = [name for name, passed in checks.items() if not passed]
            raise RuntimeError(f"A4 multimodal checks failed: {failed}")
        print(
            "PASS: "
            f"labels={len(compiled.sensor_labels)}, "
            f"closed_visible={closed_metrics['visible_instance_count']}, "
            f"open_visible={open_metrics['visible_instance_count']}, "
            f"valid_pixels={closed_metrics['valid_pixels']}",
            flush=True,
        )
        print(f"Contact sheet: {contact_path}", flush=True)
        print(f"Metrics:       {metrics_path}", flush=True)
        return 0
    finally:
        scene = None
        rr.destroy()


if __name__ == "__main__":
    raise SystemExit(main())
