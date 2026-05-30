from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


@dataclass(frozen=True)
class MeshData:
    vertices: np.ndarray
    triangles: np.ndarray
    source_backend: str


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_genesis_mesh(repo_root: Path) -> Path:
    return repo_root.parent / "Genesis" / "genesis" / "assets" / "meshes" / "duck" / "duck.obj"


def _fallback_mesh(repo_root: Path) -> Path:
    return repo_root / "external" / "donut" / "thirdparty" / "cgltf" / "fuzz" / "data" / "Box.glb"


def _resolve_default_mesh(repo_root: Path) -> Path:
    genesis_duck = _default_genesis_mesh(repo_root)
    if genesis_duck.is_file():
        return genesis_duck
    box = _fallback_mesh(repo_root)
    if box.is_file():
        return box
    raise FileNotFoundError("No default Genesis duck OBJ or bundled Box.glb mesh was found.")


def _load_with_open3d(path: Path) -> MeshData | None:
    try:
        import open3d as o3d
    except ModuleNotFoundError:
        return None

    mesh = o3d.io.read_triangle_mesh(str(path))
    if mesh.is_empty() or len(mesh.triangles) == 0:
        return None
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    triangles = np.asarray(mesh.triangles, dtype=np.int32)
    return MeshData(vertices=vertices, triangles=triangles, source_backend="open3d.io")


def _as_mesh_list(loaded) -> list:
    try:
        import trimesh
    except ModuleNotFoundError as exc:
        raise RuntimeError("trimesh is required as a fallback mesh loader.") from exc

    if isinstance(loaded, trimesh.Scene):
        meshes = []
        for geometry in loaded.geometry.values():
            if isinstance(geometry, trimesh.Trimesh) and len(geometry.faces) > 0:
                meshes.append(geometry)
        return meshes
    if isinstance(loaded, trimesh.Trimesh):
        return [loaded]
    return []


def _load_with_trimesh(path: Path) -> MeshData | None:
    try:
        import trimesh
    except ModuleNotFoundError:
        return None

    loaded = trimesh.load(str(path), force="scene" if path.suffix.lower() == ".glb" else None)
    meshes = _as_mesh_list(loaded)
    if not meshes:
        return None

    vertices_parts = []
    triangle_parts = []
    vertex_offset = 0
    for mesh in meshes:
        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        triangles = np.asarray(mesh.faces, dtype=np.int32)
        if vertices.size == 0 or triangles.size == 0:
            continue
        vertices_parts.append(vertices)
        triangle_parts.append(triangles + vertex_offset)
        vertex_offset += vertices.shape[0]

    if not vertices_parts:
        return None
    return MeshData(
        vertices=np.concatenate(vertices_parts, axis=0),
        triangles=np.concatenate(triangle_parts, axis=0),
        source_backend="trimesh",
    )


def _parse_obj(path: Path) -> MeshData:
    positions: list[tuple[float, float, float]] = []
    triangles: list[tuple[int, int, int]] = []

    def parse_index(token: str) -> int:
        raw = int(token.split("/")[0])
        if raw > 0:
            return raw - 1
        if raw < 0:
            return len(positions) + raw
        raise ValueError("OBJ indices are 1-based and cannot be zero.")

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("v "):
                _, x, y, z = line.split()[:4]
                positions.append((float(x), float(y), float(z)))
            elif line.startswith("f "):
                face = [parse_index(token) for token in line.split()[1:]]
                for corner in range(1, len(face) - 1):
                    triangles.append((face[0], face[corner], face[corner + 1]))

    if not positions or not triangles:
        raise ValueError(f"OBJ mesh has no triangles: {path}")
    return MeshData(
        vertices=np.asarray(positions, dtype=np.float32),
        triangles=np.asarray(triangles, dtype=np.int32),
        source_backend="builtin_obj_parser",
    )


def load_mesh(path: Path) -> MeshData:
    if not path.is_file():
        raise FileNotFoundError(path)

    for loader in (_load_with_open3d, _load_with_trimesh):
        mesh = loader(path)
        if mesh is not None and mesh.vertices.size > 0 and mesh.triangles.size > 0:
            return mesh

    if path.suffix.lower() == ".obj":
        return _parse_obj(path)
    raise RuntimeError(f"Failed to load mesh with Open3D or trimesh: {path}")


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
    lengths = np.maximum(lengths, 1.0e-8)
    return vectors / lengths


def _auto_sensor_origin(vertices: np.ndarray) -> np.ndarray:
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    center = 0.5 * (mins + maxs)
    extent = np.maximum(maxs - mins, 1.0e-6)
    diag = float(np.linalg.norm(extent))
    return center + np.array((0.0, 0.20 * diag, -1.55 * diag), dtype=np.float32)


def _auto_max_range(vertices: np.ndarray, origin: np.ndarray) -> float:
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    corners = np.array(
        [
            (mins[0], mins[1], mins[2]),
            (mins[0], mins[1], maxs[2]),
            (mins[0], maxs[1], mins[2]),
            (mins[0], maxs[1], maxs[2]),
            (maxs[0], mins[1], mins[2]),
            (maxs[0], mins[1], maxs[2]),
            (maxs[0], maxs[1], mins[2]),
            (maxs[0], maxs[1], maxs[2]),
        ],
        dtype=np.float32,
    )
    return float(np.max(np.linalg.norm(corners - origin.reshape(1, 3), axis=1)) * 1.10)


def _parse_vec3(value: str | None, fallback: np.ndarray) -> np.ndarray:
    if value is None:
        return fallback.astype(np.float32)
    parts = [float(part.strip()) for part in value.split(",")]
    if len(parts) != 3:
        raise ValueError("Expected a comma-separated vector: x,y,z")
    return np.asarray(parts, dtype=np.float32)


def make_lidar_rays(
    *,
    origin: np.ndarray,
    channels: int,
    horizontal_steps: int,
    vertical_min_deg: float,
    vertical_max_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    azimuths = np.linspace(0.0, 2.0 * math.pi, horizontal_steps, endpoint=False, dtype=np.float32)
    elevations = np.linspace(
        math.radians(vertical_min_deg),
        math.radians(vertical_max_deg),
        channels,
        dtype=np.float32,
    )
    azimuth_grid, elevation_grid = np.meshgrid(azimuths, elevations)
    cos_el = np.cos(elevation_grid)
    directions = np.stack(
        (
            cos_el * np.sin(azimuth_grid),
            np.sin(elevation_grid),
            cos_el * np.cos(azimuth_grid),
        ),
        axis=-1,
    ).reshape(-1, 3)
    directions = _normalize_rows(directions.astype(np.float32))
    origins = np.repeat(origin.reshape(1, 3), directions.shape[0], axis=0).astype(np.float32)
    return origins, directions


def _look_at_basis(origin: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    forward = target - origin
    forward = forward / max(float(np.linalg.norm(forward)), 1.0e-8)
    world_up = np.array((0.0, 1.0, 0.0), dtype=np.float32)
    right = np.cross(world_up, forward)
    if float(np.linalg.norm(right)) <= 1.0e-6:
        right = np.array((1.0, 0.0, 0.0), dtype=np.float32)
    right = right / max(float(np.linalg.norm(right)), 1.0e-8)
    up = np.cross(forward, right)
    up = up / max(float(np.linalg.norm(up)), 1.0e-8)
    return forward.astype(np.float32), right.astype(np.float32), up.astype(np.float32)


def make_focused_rays(
    *,
    origin: np.ndarray,
    target: np.ndarray,
    channels: int,
    horizontal_steps: int,
    vertical_min_deg: float,
    vertical_max_deg: float,
    horizontal_fov_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    forward, right, up = _look_at_basis(origin, target)
    yaws = np.linspace(
        -0.5 * math.radians(horizontal_fov_deg),
        0.5 * math.radians(horizontal_fov_deg),
        horizontal_steps,
        dtype=np.float32,
    )
    pitches = np.linspace(
        math.radians(vertical_min_deg),
        math.radians(vertical_max_deg),
        channels,
        dtype=np.float32,
    )
    yaw_grid, pitch_grid = np.meshgrid(yaws, pitches)
    directions = (
        forward.reshape(1, 1, 3)
        + np.tan(yaw_grid).reshape(channels, horizontal_steps, 1) * right.reshape(1, 1, 3)
        + np.tan(pitch_grid).reshape(channels, horizontal_steps, 1) * up.reshape(1, 1, 3)
    ).reshape(-1, 3)
    directions = _normalize_rows(directions.astype(np.float32))
    origins = np.repeat(origin.reshape(1, 3), directions.shape[0], axis=0).astype(np.float32)
    return origins, directions


def make_orbit_rays(
    *,
    vertices: np.ndarray,
    channels: int,
    horizontal_steps: int,
    vertical_min_deg: float,
    vertical_max_deg: float,
    horizontal_fov_deg: float,
    orbit_poses: int,
) -> tuple[np.ndarray, np.ndarray, list[list[float]]]:
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    center = 0.5 * (mins + maxs)
    extent = np.maximum(maxs - mins, 1.0e-6)
    radius = float(np.linalg.norm(extent[[0, 2]])) * 1.15
    radius = max(radius, float(np.max(extent)) * 1.05)
    target = center + np.array((0.0, 0.05 * extent[1], 0.0), dtype=np.float32)
    height = center[1] + 0.06 * extent[1]

    all_origins = []
    all_directions = []
    sensor_origins: list[list[float]] = []
    for index in range(max(1, orbit_poses)):
        angle = 2.0 * math.pi * index / max(1, orbit_poses)
        origin = np.array(
            (
                center[0] + radius * math.sin(angle),
                height,
                center[2] - radius * math.cos(angle),
            ),
            dtype=np.float32,
        )
        origins, directions = make_focused_rays(
            origin=origin,
            target=target,
            channels=channels,
            horizontal_steps=horizontal_steps,
            vertical_min_deg=vertical_min_deg,
            vertical_max_deg=vertical_max_deg,
            horizontal_fov_deg=horizontal_fov_deg,
        )
        all_origins.append(origins)
        all_directions.append(directions)
        sensor_origins.append([float(v) for v in origin])

    return np.concatenate(all_origins, axis=0), np.concatenate(all_directions, axis=0), sensor_origins


def raycast_open3d(mesh: MeshData, origins: np.ndarray, directions: np.ndarray) -> tuple[np.ndarray, str]:
    import open3d as o3d

    scene = o3d.t.geometry.RaycastingScene()
    vertices = o3d.core.Tensor(mesh.vertices, dtype=o3d.core.Dtype.Float32)
    triangles = o3d.core.Tensor(mesh.triangles.astype(np.uint32), dtype=o3d.core.Dtype.UInt32)
    scene.add_triangles(vertices, triangles)
    rays = np.concatenate((origins, directions), axis=1).astype(np.float32)
    result = scene.cast_rays(o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32))
    return result["t_hit"].numpy().astype(np.float32), "open3d.RaycastingScene"


def raycast_numpy(mesh: MeshData, origins: np.ndarray, directions: np.ndarray) -> tuple[np.ndarray, str]:
    if origins.shape[0] > 50000:
        raise RuntimeError(
            "The numpy fallback is intended for small smoke tests. Install Open3D or reduce "
            "--channels, --horizontal-steps, or --orbit-poses for this run."
        )

    vertices = mesh.vertices
    triangles = mesh.triangles
    t_hit = np.full((origins.shape[0],), np.inf, dtype=np.float32)
    eps = 1.0e-7

    for tri in vertices[triangles]:
        v0, v1, v2 = tri
        edge1 = v1 - v0
        edge2 = v2 - v0
        pvec = np.cross(directions, edge2)
        det = np.einsum("ij,j->i", pvec, edge1)
        mask = np.abs(det) > eps
        if not np.any(mask):
            continue
        inv_det = np.zeros_like(det)
        inv_det[mask] = 1.0 / det[mask]
        tvec = origins - v0
        u = np.einsum("ij,ij->i", tvec, pvec) * inv_det
        mask &= (u >= 0.0) & (u <= 1.0)
        qvec = np.cross(tvec, edge1)
        v = np.einsum("ij,ij->i", directions, qvec) * inv_det
        mask &= (v >= 0.0) & ((u + v) <= 1.0)
        t = np.einsum("j,ij->i", edge2, qvec) * inv_det
        mask &= (t > eps) & (t < t_hit)
        t_hit[mask] = t[mask]

    return t_hit, "numpy_moller_trumbore"


def _write_point_cloud(path: Path, points: np.ndarray, ranges: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if points.size:
        denom = max(float(np.max(ranges) - np.min(ranges)), 1.0e-6)
        normalized = (ranges - float(np.min(ranges))) / denom
        colors = np.stack(
            (
                80.0 + 120.0 * (1.0 - normalized),
                170.0 + 70.0 * normalized,
                210.0 + 35.0 * normalized,
            ),
            axis=1,
        ).astype(np.uint8)
    else:
        colors = np.zeros((0, 3), dtype=np.uint8)

    with path.open("w", encoding="ascii") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for point, color in zip(points, colors):
            f.write(
                f"{point[0]:.7f} {point[1]:.7f} {point[2]:.7f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def _write_range_image(path: Path, ranges: np.ndarray, valid: np.ndarray, max_range: float) -> None:
    image = np.zeros(ranges.shape, dtype=np.uint8)
    normalized = 1.0 - np.clip(ranges / max_range, 0.0, 1.0)
    image[valid] = np.clip(normalized[valid] * 255.0, 0.0, 255.0).astype(np.uint8)
    Image.fromarray(image).save(path)


def _write_bev(path: Path, points: np.ndarray, origin: np.ndarray, max_range: float, size: int = 768) -> None:
    image = Image.new("RGB", (size, size), (8, 10, 14))
    draw = ImageDraw.Draw(image)
    center = size // 2
    scale = (size * 0.46) / max(max_range, 1.0e-6)

    for radius in (0.25, 0.5, 0.75, 1.0):
        r = int(radius * max_range * scale)
        draw.ellipse((center - r, center - r, center + r, center + r), outline=(36, 48, 60))

    if points.size:
        offsets = points[:, [0, 2]] - origin[[0, 2]]
        pixels = np.empty_like(offsets)
        pixels[:, 0] = center + offsets[:, 0] * scale
        pixels[:, 1] = center - offsets[:, 1] * scale
        pixels = np.rint(pixels).astype(np.int32)
        inside = (
            (pixels[:, 0] >= 0)
            & (pixels[:, 0] < size)
            & (pixels[:, 1] >= 0)
            & (pixels[:, 1] < size)
        )
        for x, y in pixels[inside]:
            draw.point((int(x), int(y)), fill=(75, 220, 185))

    draw.ellipse((center - 4, center - 4, center + 4, center + 4), fill=(250, 220, 90))
    image.save(path)


def _point_colors(ranges: np.ndarray) -> np.ndarray:
    if ranges.size == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    denom = max(float(np.max(ranges) - np.min(ranges)), 1.0e-6)
    normalized = (ranges - float(np.min(ranges))) / denom
    return np.stack(
        (
            90.0 + 140.0 * (1.0 - normalized),
            170.0 + 70.0 * normalized,
            215.0 + 30.0 * normalized,
        ),
        axis=1,
    ).astype(np.uint8)


def _write_projection(
    path: Path,
    coords: np.ndarray,
    ranges: np.ndarray,
    *,
    size: int = 768,
    point_radius: int = 1,
) -> None:
    image = Image.new("RGB", (size, size), (8, 10, 14))
    draw = ImageDraw.Draw(image)
    if coords.size == 0:
        image.save(path)
        return

    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    center = 0.5 * (mins + maxs)
    extent = np.maximum(maxs - mins, 1.0e-6)
    margin = 38
    scale = (size - margin * 2) / float(np.max(extent))

    pixels = np.empty_like(coords, dtype=np.float32)
    pixels[:, 0] = size * 0.5 + (coords[:, 0] - center[0]) * scale
    pixels[:, 1] = size * 0.5 - (coords[:, 1] - center[1]) * scale
    pixels = np.rint(pixels).astype(np.int32)
    colors = _point_colors(ranges)

    # Sort far-to-near by range so closer points stay visible in dense projections.
    order = np.argsort(ranges)[::-1] if ranges.size else np.arange(pixels.shape[0])
    for idx in order:
        x, y = pixels[idx]
        if x < 0 or x >= size or y < 0 or y >= size:
            continue
        color = tuple(int(v) for v in colors[idx])
        if point_radius <= 1:
            draw.point((int(x), int(y)), fill=color)
        else:
            draw.ellipse(
                (int(x - point_radius), int(y - point_radius), int(x + point_radius), int(y + point_radius)),
                fill=color,
            )

    image.save(path)


def _write_view_images(output_dir: Path, points: np.ndarray, ranges: np.ndarray) -> dict[str, Path]:
    top_path = output_dir / "bev_top.png"
    front_path = output_dir / "bev_front.png"
    side_path = output_dir / "bev_side.png"
    iso_path = output_dir / "bev_iso.png"

    _write_projection(top_path, points[:, [0, 2]], ranges)
    _write_projection(front_path, points[:, [0, 1]], ranges)
    _write_projection(side_path, points[:, [2, 1]], ranges)

    iso_x = (points[:, 0] - points[:, 2]) * 0.70710678
    iso_y = points[:, 1] + (points[:, 0] + points[:, 2]) * 0.23
    _write_projection(iso_path, np.stack((iso_x, iso_y), axis=1), ranges)
    return {
        "bev_top": top_path,
        "bev_front": front_path,
        "bev_side": side_path,
        "bev_iso": iso_path,
    }


def _range_stats(ranges: np.ndarray) -> dict[str, float | None]:
    if ranges.size == 0:
        return {"min_range": None, "max_observed_range": None, "mean_range": None}
    return {
        "min_range": float(np.min(ranges)),
        "max_observed_range": float(np.max(ranges)),
        "mean_range": float(np.mean(ranges)),
    }


def main() -> int:
    repo_root = _repo_root()
    parser = argparse.ArgumentParser(description="Generate a LiDAR-like sidecar smoke test from a mesh.")
    parser.add_argument("--mesh", type=Path, default=None, help="Input .glb/.obj/.ply mesh.")
    parser.add_argument("--output-dir", type=Path, default=repo_root / "outputs" / "lidar_smoke")
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--horizontal-steps", type=int, default=1024)
    parser.add_argument("--vertical-min", type=float, default=-25.0)
    parser.add_argument("--vertical-max", type=float, default=25.0)
    parser.add_argument("--horizontal-fov", type=float, default=70.0)
    parser.add_argument("--scan-mode", choices=("single", "object_orbit"), default="object_orbit")
    parser.add_argument("--orbit-poses", type=int, default=12)
    parser.add_argument("--max-range", type=float, default=0.0, help="Meters/scene units. Use 0 for auto.")
    parser.add_argument("--sensor-origin", type=str, default=None, help="Optional x,y,z sensor origin.")
    parser.add_argument(
        "--backend",
        choices=("auto", "open3d", "numpy"),
        default="auto",
        help="Raycast backend. auto prefers Open3D RaycastingScene.",
    )
    args = parser.parse_args()

    mesh_path = (args.mesh or _resolve_default_mesh(repo_root)).resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    mesh = load_mesh(mesh_path)
    if args.scan_mode == "object_orbit":
        origins, directions, sensor_origins = make_orbit_rays(
            vertices=mesh.vertices,
            channels=max(1, args.channels),
            horizontal_steps=max(1, args.horizontal_steps),
            vertical_min_deg=args.vertical_min,
            vertical_max_deg=args.vertical_max,
            horizontal_fov_deg=args.horizontal_fov,
            orbit_poses=max(1, args.orbit_poses),
        )
        origin_for_bev = np.mean(np.asarray(sensor_origins, dtype=np.float32), axis=0)
    else:
        origin = _parse_vec3(args.sensor_origin, _auto_sensor_origin(mesh.vertices))
        origins, directions = make_lidar_rays(
            origin=origin,
            channels=max(1, args.channels),
            horizontal_steps=max(1, args.horizontal_steps),
            vertical_min_deg=args.vertical_min,
            vertical_max_deg=args.vertical_max,
        )
        sensor_origins = [[float(v) for v in origin]]
        origin_for_bev = origin

    raycast_backend = ""
    if args.backend in ("auto", "open3d"):
        try:
            t_hit, raycast_backend = raycast_open3d(mesh, origins, directions)
        except ModuleNotFoundError:
            if args.backend == "open3d":
                raise
            t_hit, raycast_backend = raycast_numpy(mesh, origins, directions)
    else:
        t_hit, raycast_backend = raycast_numpy(mesh, origins, directions)

    max_range = float(args.max_range)
    if max_range <= 0.0:
        max_range = max(
            _auto_max_range(mesh.vertices, np.asarray(sensor_origin, dtype=np.float32))
            for sensor_origin in sensor_origins
        )
    valid_flat = np.isfinite(t_hit) & (t_hit > 0.0) & (t_hit <= max_range)
    valid_ranges = t_hit[valid_flat]
    points = origins[valid_flat] + directions[valid_flat] * valid_ranges.reshape(-1, 1)

    image_width = max(1, args.horizontal_steps)
    if args.scan_mode == "object_orbit":
        pose_count = max(1, args.orbit_poses)
        range_grid = (
            t_hit.reshape(pose_count, max(1, args.channels), image_width)
            .transpose(1, 0, 2)
            .reshape(max(1, args.channels), pose_count * image_width)
        )
        valid_grid = (
            valid_flat.reshape(pose_count, max(1, args.channels), image_width)
            .transpose(1, 0, 2)
            .reshape(max(1, args.channels), pose_count * image_width)
        )
    else:
        range_grid = t_hit.reshape(max(1, args.channels), image_width)
        valid_grid = valid_flat.reshape(max(1, args.channels), image_width)
    display_ranges = np.where(valid_grid, range_grid, max_range)

    points_path = output_dir / "lidar_points.ply"
    range_image_path = output_dir / "range_image.png"
    bev_path = output_dir / "bev.png"
    stats_path = output_dir / "lidar_stats.json"

    _write_point_cloud(points_path, points, valid_ranges)
    _write_range_image(range_image_path, display_ranges, valid_grid, max_range)
    _write_bev(bev_path, points, origin_for_bev, max_range)
    view_paths = _write_view_images(output_dir, points, valid_ranges)

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    stats = {
        "mesh_path": str(mesh_path),
        "mesh_loader": mesh.source_backend,
        "raycast_backend": raycast_backend,
        "ray_count": int(t_hit.size),
        "valid_hit_count": int(valid_ranges.size),
        "hit_ratio": float(valid_ranges.size / max(1, t_hit.size)),
        **_range_stats(valid_ranges),
        "channels": int(max(1, args.channels)),
        "horizontal_steps": int(max(1, args.horizontal_steps)),
        "scan_mode": str(args.scan_mode),
        "orbit_poses": int(max(1, args.orbit_poses)) if args.scan_mode == "object_orbit" else 1,
        "horizontal_fov_degrees": float(args.horizontal_fov if args.scan_mode == "object_orbit" else 360.0),
        "vertical_fov_degrees": [float(args.vertical_min), float(args.vertical_max)],
        "max_range": max_range,
        "sensor_origin": sensor_origins[0],
        "sensor_origins": sensor_origins,
        "runtime_ms": float(elapsed_ms),
        "outputs": {
            "point_cloud": str(points_path),
            "range_image": str(range_image_path),
            "bev": str(bev_path),
            **{name: str(path) for name, path in view_paths.items()},
        },
        "limitations": [
            "sidecar mesh raycasting only",
            "no intensity",
            "no noise model",
            "no motion distortion",
            "no semantic label",
            "not integrated into RTXNS native renderer",
        ],
    }
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
