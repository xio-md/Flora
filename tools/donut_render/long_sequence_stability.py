from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np


def _default_module_dir(repo_root: Path) -> Path:
    platform_dir = "windows-x64" if os.name == "nt" else "linux-x64"
    return repo_root / "bin" / platform_dir


def _grid_sheet_mesh(nx: int, ny: int, phase: float) -> tuple[np.ndarray, np.ndarray]:
    """Fixed topology grid in XZ plane; Y wiggles with phase (deformable stress test)."""
    nx = max(2, int(nx))
    ny = max(2, int(ny))
    verts: list[tuple[float, float, float]] = []
    for j in range(ny):
        for i in range(nx):
            u = i / (nx - 1)
            v = j / (ny - 1)
            x = (u - 0.5) * 2.2
            z = (v - 0.5) * 2.2
            y = 0.12 * math.sin(phase + u * 6.2 + v * 4.1) + 0.02 * math.sin(phase * 1.7 + (i + j) * 0.3)
            verts.append((x, y, z))
    vertices = np.asarray(verts, dtype=np.float32)
    tris: list[tuple[int, int, int]] = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            i0 = j * nx + i
            i1 = i0 + 1
            i2 = i0 + nx
            i3 = i2 + 1
            tris.append((i0, i2, i1))
            tris.append((i1, i2, i3))
    triangles = np.asarray(tris, dtype=np.uint32)
    return vertices, triangles


def _particle_layout(frame: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    centers = []
    radii = []
    for k in range(n):
        ang = (k / max(1, n)) * math.tau + 0.08 * frame
        r_orbit = 0.55 + 0.12 * math.sin(frame * 0.11 + k)
        centers.append((r_orbit * math.cos(ang), 0.18 + 0.04 * math.sin(frame * 0.19 + k), r_orbit * math.sin(ang)))
        radii.append(0.035 + 0.018 * abs(math.sin(frame * 0.23 + k * 0.7)))
    return np.asarray(centers, dtype=np.float32), np.asarray(radii, dtype=np.float32)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Long-frame DonutRenderPy run: deformable grid + particles each frame (stability harness)."
    )
    parser.add_argument("--module-dir", type=Path, default=_default_module_dir(repo_root))
    parser.add_argument("--runtime-dir", type=Path, default=repo_root)
    parser.add_argument(
        "--frames",
        type=int,
        default=120,
        help="Number of frames (default 120; use 100+ for long-run coverage).",
    )
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / ".temp" / "long_sequence_stability.json",
        help="JSON summary path.",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    frame_count = max(1, int(args.frames))
    if frame_count < 100:
        print("warning: fewer than 100 frames; long-run coverage is reduced.", file=sys.stderr)

    sys.path.insert(0, str(repo_root / "python"))
    import DonutRenderPy as dr

    width = max(1, int(args.width))
    height = max(1, int(args.height))
    expected_rgba = width * height * 4

    ground_verts = np.array(
        [
            (-3.0, 0.0, -3.0),
            (3.0, 0.0, -3.0),
            (3.0, 0.0, 3.0),
            (-3.0, 0.0, 3.0),
        ],
        dtype=np.float32,
    )
    ground_tris = np.array([(0, 2, 1), (0, 3, 2)], dtype=np.uint32)

    dr.init(
        context_path=repo_root,
        runtime_dir=args.runtime_dir,
        module_dir=args.module_dir,
        backend="vulkan",
        device_index=-1,
        log_level=dr.LogLevel.WARNING,
    )

    frames_out: list[dict[str, object]] = []
    ok = True
    scene = None
    try:
        scene = dr.create_scene()
        scene.init(
            dr.Render(
                name="dynamic-long-sequence",
                spectrum=dr.SRGBSpectrum(),
                integrator=dr.WavePathIntegrator(log_level=dr.LogLevel.WARNING, max_depth=4),
            )
        )
        scene.update_environment(dr.Environment(name="env", emission=dr.ColorTexture((0.08, 0.09, 0.11))))

        surf_ground = dr.PlasticSurface(name="ground", kd=dr.ColorTexture((0.55, 0.58, 0.62, 1.0)), roughness=dr.ColorTexture((0.92,)))
        surf_sheet = dr.PlasticSurface(name="sheet", kd=dr.ColorTexture((0.75, 0.35, 0.22, 1.0)), roughness=dr.ColorTexture((0.45,)))
        surf_dots = dr.PlasticSurface(name="dots", kd=dr.ColorTexture((0.85, 0.9, 1.0, 1.0)), roughness=dr.ColorTexture((0.2,)))

        scene.update_surface(surf_ground)
        scene.update_surface(surf_sheet)
        scene.update_surface(surf_dots)

        ground = dr.RigidShape(name="ground", vertices=ground_verts, triangles=ground_tris, surface=surf_ground)
        v0, t0 = _grid_sheet_mesh(5, 5, 0.0)
        sheet = dr.DeformableShape(name="sheet", vertices=v0, triangles=t0, surface=surf_sheet)
        c0, r0 = _particle_layout(0.0, 10)
        dots = dr.ParticlesShape(name="dots", centers=c0, radii=r0, surface=surf_dots)

        camera = dr.PinholeCamera(
            name="cam",
            pose=dr.MatrixTransform(
                (
                    (1.0, 0.0, 0.0, 0.0),
                    (0.0, 1.0, 0.0, 0.0),
                    (0.0, 0.0, 1.0, 2.8),
                    (0.0, 0.0, 0.0, 1.0),
                )
            ),
            film=dr.Film((width, height)),
            filter=dr.Filter(1.0),
            spp=1,
            fov=50.0,
        )

        scene.update_shape(ground)
        scene.update_shape(sheet)
        scene.update_shape(dots)
        scene.update_camera(camera, denoise=False)

        for frame_index in range(frame_count):
            t = float(frame_index)
            v1, _ = _grid_sheet_mesh(5, 5, t * 0.15)
            sheet.update(v1, t0)
            c1, r1 = _particle_layout(t, 10)
            dots.update(c1, r1)

            scene.update_shape(sheet)
            scene.update_shape(dots)

            t0_update = time.perf_counter()
            scene.update_scene(time=t)
            update_ms = (time.perf_counter() - t0_update) * 1000.0

            t0_render = time.perf_counter()
            rgba = scene.render_frame(camera)
            render_ms = (time.perf_counter() - t0_render) * 1000.0

            if len(rgba) != expected_rgba:
                ok = False
            frames_out.append(
                {
                    "frame": frame_index,
                    "time": t,
                    "update_scene_ms": float(update_ms),
                    "render_frame_ms": float(render_ms),
                    "rgba_bytes": int(len(rgba)),
                    "ok": bool(len(rgba) == expected_rgba),
                }
            )

        scene.destroy()
        scene = None
    finally:
        if scene is not None:
            scene.destroy()
        dr.destroy()

    summary = {
        "tool": "long_sequence_stability",
        "frame_count": frame_count,
        "resolution": [width, height],
        "expected_rgba_bytes": expected_rgba,
        "all_frames_ok": ok and all(bool(f["ok"]) for f in frames_out),
        "max_update_scene_ms": max((float(f["update_scene_ms"]) for f in frames_out), default=0.0),
        "max_render_frame_ms": max((float(f["render_frame_ms"]) for f in frames_out), default=0.0),
        "total_update_scene_ms": float(sum(float(f["update_scene_ms"]) for f in frames_out)),
        "total_render_frame_ms": float(sum(float(f["render_frame_ms"]) for f in frames_out)),
        "frames": frames_out,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if not args.quiet:
        print(json.dumps({k: summary[k] for k in summary if k != "frames"}, indent=2))
        print(str(args.output))
    return 0 if summary["all_frames_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
