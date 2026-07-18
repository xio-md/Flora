"""Validate async readback ring occupancy and frame integrity."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENE = REPO_ROOT / "ReplicaCAD" / "stages" / "frl_apartment_stage.glb"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--ring-depth", type=int, default=4)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=144)
    parser.add_argument("--rt-shadows", action="store_true")
    return parser.parse_args()


def frame_hash(frame: bytes) -> str:
    return hashlib.sha256(frame).hexdigest()


def main() -> int:
    args = parse_args()
    if args.ring_depth < 2:
        raise ValueError("--ring-depth must be at least 2")
    if not args.scene.is_file():
        raise FileNotFoundError(f"ReplicaCAD scene not found: {args.scene}")

    module_dir = REPO_ROOT / "bin" / "windows-x64"
    sys.path.insert(0, str(module_dir))
    import DonutRenderPyNative as rr

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

        if scene.render_frame_batch([]) != []:
            print("FAIL: an empty synchronous batch did not return an empty list", flush=True)
            return 1
        print("  empty synchronous batch: verified", flush=True)

        camera = scene.add_camera(
            position=[3.5, 2.0, 3.5],
            target=[0.0, 1.0, 0.0],
            up=[0.0, 1.0, 0.0],
            fov_degrees=60,
            width=args.width,
            height=args.height,
        )

        try:
            scene.submit_frame_batch([camera, camera])
        except ValueError:
            print("  duplicate camera indices: rejected", flush=True)
        else:
            print("FAIL: duplicate camera indices were accepted", flush=True)
            return 1

        scene.set_camera_at(
            0,
            position=[-3.0, 2.5, 2.5],
            target=[0.0, 1.0, 0.0],
            up=[0.0, 1.0, 0.0],
            fov_degrees=60,
            width=args.width,
            height=args.height,
        )
        camera_zero_reference = frame_hash(scene.render_frame(0))
        scene.render_frame(camera)
        if frame_hash(scene.render_frame(0)) != camera_zero_reference:
            print("FAIL: camera 0 state was not restored after rendering another camera", flush=True)
            return 1
        print("  legacy camera 1 -> 0 state restore: verified", flush=True)

        frame_count = args.ring_depth + 2
        references: dict[int, str] = {}
        print(
            f"Ring correctness: scene={args.scene.name}, "
            f"resolution={args.width}x{args.height}, K={args.ring_depth}, "
            f"RT={args.rt_shadows}",
            flush=True,
        )

        for frame_id in range(frame_count):
            scene.set_camera_at(
                camera,
                position=[3.5 + frame_id * 0.08, 2.0, 3.5],
                target=[0.0, 1.0, 0.0],
                up=[0.0, 1.0, 0.0],
                fov_degrees=60,
                width=args.width,
                height=args.height,
            )
            references[frame_id] = frame_hash(scene.render_frame(camera))

        submitted: list[tuple[int, int]] = []
        busy_count = 0
        for frame_id in range(frame_count):
            scene.set_camera_at(
                camera,
                position=[3.5 + frame_id * 0.08, 2.0, 3.5],
                target=[0.0, 1.0, 0.0],
                up=[0.0, 1.0, 0.0],
                fov_degrees=60,
                width=args.width,
                height=args.height,
            )
            token = scene.submit_frame_batch([camera])
            if token == 0:
                busy_count += 1
                print(f"  submit frame {frame_id}: BUSY", flush=True)
            else:
                submitted.append((frame_id, token))
                print(
                    f"  submit frame {frame_id}: token={token}, "
                    f"ready_immediately={scene.is_batch_ready(token)}",
                    flush=True,
                )

        errors = 0
        for frame_id, token in reversed(submitted):
            frames = scene.read_frame_batch(token)
            actual = frame_hash(frames[0])
            expected = references[frame_id]
            ok = actual == expected
            errors += int(not ok)
            print(f"  reverse read frame {frame_id}: {'OK' if ok else 'CORRUPTED'}", flush=True)

        expected_submitted = args.ring_depth
        expected_busy = frame_count - args.ring_depth
        if len(submitted) != expected_submitted or busy_count != expected_busy:
            print(
                f"FAIL: submitted={len(submitted)} (expected {expected_submitted}), "
                f"busy={busy_count} (expected {expected_busy})",
                flush=True,
            )
            return 1
        if errors:
            print(f"FAIL: {errors} frame hashes mismatched", flush=True)
            return 1

        print(
            f"PASS: {len(submitted)} frames verified in reverse order; "
            f"{busy_count} over-capacity submits rejected",
            flush=True,
        )
        return 0
    finally:
        rr.destroy()


if __name__ == "__main__":
    raise SystemExit(main())
