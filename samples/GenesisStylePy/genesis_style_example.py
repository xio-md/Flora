from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from python_demo_common import default_output_dir, frame_output_path, write_rgb_ppm


def _make_box(size: tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sx, sy, sz = (0.5 * float(v) for v in size)
    face_vertices = [
        [(-sx, -sy, -sz), (-sx, sy, -sz), (sx, sy, -sz), (sx, -sy, -sz)],
        [(-sx, -sy, sz), (sx, -sy, sz), (sx, sy, sz), (-sx, sy, sz)],
        [(-sx, -sy, -sz), (sx, -sy, -sz), (sx, -sy, sz), (-sx, -sy, sz)],
        [(sx, -sy, -sz), (sx, sy, -sz), (sx, sy, sz), (sx, -sy, sz)],
        [(sx, sy, -sz), (-sx, sy, -sz), (-sx, sy, sz), (sx, sy, sz)],
        [(-sx, sy, -sz), (-sx, -sy, -sz), (-sx, -sy, sz), (-sx, sy, sz)],
    ]
    face_normals = [
        (0.0, 0.0, -1.0),
        (0.0, 0.0, 1.0),
        (0.0, -1.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (-1.0, 0.0, 0.0),
    ]

    vertices = np.array(
        [vertex for face in face_vertices for vertex in face],
        dtype=np.float32,
    )
    normals = np.array(
        [normal for normal in face_normals for _ in range(4)],
        dtype=np.float32,
    )
    triangles = np.array(
        [
            (0, 1, 2),
            (0, 2, 3),
            (4, 5, 6),
            (4, 6, 7),
            (8, 9, 10),
            (8, 10, 11),
            (12, 13, 14),
            (12, 14, 15),
            (16, 17, 18),
            (16, 18, 19),
            (20, 21, 22),
            (20, 22, 23),
        ],
        dtype=np.uint32,
    )
    return vertices, triangles, normals


def _make_plane(size: float) -> tuple[np.ndarray, np.ndarray]:
    half = 0.5 * float(size)
    vertices = np.array(
        [
            (-half, 0.0, -half),
            (half, 0.0, -half),
            (half, 0.0, half),
            (-half, 0.0, half),
        ],
        dtype=np.float32,
    )
    triangles = np.array(
        [
            (0, 2, 1),
            (0, 3, 2),
        ],
        dtype=np.uint32,
    )
    return vertices, triangles


def _rotation_y(angle_degrees: float) -> np.ndarray:
    angle = math.radians(angle_degrees)
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array(
        [
            [c, 0.0, s, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-s, 0.0, c, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser(description="Render one Genesis-style frame through the Donut Python backend.")
    parser.add_argument("--output", type=Path, default=None, help="Legacy direct output path. Overrides --output-dir/output-stem.")
    parser.add_argument("--output-dir", type=Path, default=default_output_dir(repo_root, "genesis_style_py"))
    parser.add_argument("--output-stem", type=str, default="genesis_style_frame")
    parser.add_argument("--module-dir", type=Path, default=repo_root / "bin" / "windows-x64")
    parser.add_argument("--runtime-dir", type=Path, default=repo_root)
    parser.add_argument("--frames", type=int, default=1)
    args = parser.parse_args()

    sys.path.insert(0, str(repo_root / "python"))

    from rtxns_genesis_style import CameraDesc, GenesisStyleRenderer, SurfaceDesc

    ground_vertices, ground_triangles = _make_plane(6.0)
    box_vertices, box_triangles, box_normals = _make_box((1.0, 1.0, 1.0))
    particle_offsets = np.array(
        [
            (-0.42, 0.12, 0.25),
            (-0.18, 0.16, 0.08),
            (0.14, 0.14, -0.10),
            (0.36, 0.11, -0.28),
        ],
        dtype=np.float32,
    )
    frame_count = max(1, args.frames)

    with GenesisStyleRenderer(module_dir=args.module_dir, runtime_dir=args.runtime_dir) as renderer:
        renderer.set_ambient((0.12, 0.11, 0.10), (0.08, 0.07, 0.06))
        renderer.set_default_light(direction=(-0.7, -1.0, -0.85), color=(1.0, 0.95, 0.88), irradiance=0.9)

        renderer.add_surface("ground", SurfaceDesc(base_color=(0.68, 0.70, 0.75, 1.0), roughness=0.95))
        renderer.add_rigid("ground", ground_vertices, ground_triangles)

        renderer.add_surface(
            "box",
            SurfaceDesc(
                base_color=(0.91, 0.50, 0.18, 1.0),
                roughness=0.62,
                metallic=0.0,
                emissive=(0.10, 0.05, 0.02),
            ),
        )
        renderer.add_rigid("box", box_vertices, box_triangles, normals=box_normals)

        renderer.add_surface(
            "particles",
            SurfaceDesc(
                base_color=(0.9, 0.94, 1.0, 1.0),
                roughness=0.08,
                metallic=0.0,
            ),
        )
        renderer.add_particles("particles", radius=0.08)

        camera = CameraDesc(
            uid="main",
            pos=(2.6, 1.9, 2.9),
            lookat=(0.0, 0.45, 0.0),
            up=(0.0, 1.0, 0.0),
            res=(640, 480),
            fov=45.0,
            near=0.1,
            far=100.0,
        )
        renderer.add_camera(camera)

        last_output = None
        for frame_index in range(frame_count):
            angle = 20.0 + frame_index * 10.0
            transform = _rotation_y(angle)
            transform[:3, 3] = np.array([0.0, 0.5, 0.0], dtype=np.float32)
            renderer.update_rigid("box", transform)

            orbit_angle = math.radians(frame_index * 18.0)
            centers = particle_offsets.copy()
            centers[:, 0] += 0.18 * math.cos(orbit_angle)
            centers[:, 2] += 0.18 * math.sin(orbit_angle)
            centers[:, 1] += 0.05 * np.sin(orbit_angle + np.linspace(0.0, math.pi, centers.shape[0]))
            renderer.update_particles("particles", centers, radius=0.08)

            image = renderer.render_camera(camera, force_render=True, time=float(frame_index))
            if args.output is not None:
                last_output = args.output if frame_count <= 1 else args.output.with_name(
                    f"{args.output.stem}_{frame_index:03d}{args.output.suffix}"
                )
            else:
                last_output = frame_output_path(args.output_dir, args.output_stem, frame_index)
            write_rgb_ppm(last_output, image)

    print(last_output)
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
