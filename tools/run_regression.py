"""Automated regression test for the RTXNS ray-traced shadow pipeline.

Runs two fixed scenes (Genesis box + Bistro) and asserts:
  - Output images are generated.
  - Darkened-pixel ratio falls within expected bounds.
  - Non-shadowed regions preserve color (within tolerance).
  - GPU performance stats are collected and printed.

Usage:
    python tools/run_regression.py [--skip-bistro]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

repo_root = Path(r"D:\RTXNS")
sys.path.insert(0, str(repo_root / "bin" / "windows-x64"))
sys.path.insert(0, str(repo_root / "python"))

REPORT_DIR = repo_root / "output" / "report"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# --- Regression thresholds -------------------------------------------------
# (min_dark_ratio, max_dark_ratio) for the darkened-pixel percentage.
# Darkened = pixel where (no_shadow - rt_shadow).mean(axis=2) > DARK_THRESHOLD.
DARK_THRESHOLD = 10.0  # per-channel mean difference

GENESIS_EXPECTED = (0.1, 15.0)   # Genesis: small localized shadow (black background inflates visible area)
BISTRO_EXPECTED  = (20.0, 70.0)  # Bistro: large shadowed area

# Max allowed per-channel difference in non-shadowed regions.
# Note: RT-shadow frames go through the composite pass (filmic tonemap + exposure),
# while no-shadow frames may bypass composite on the first run, so a moderate
# difference is expected in HDR highlight regions.
NON_SHADOW_MAX_DIFF = 30


def _dark_ratio(arr_no: np.ndarray, arr_rt: np.ndarray) -> float:
    """Fraction of visible pixels that got significantly darker."""
    diff = arr_no.astype(np.float32) - arr_rt.astype(np.float32)
    dark_mask = diff.mean(axis=2) > DARK_THRESHOLD
    visible = (arr_no.sum(axis=2) > 10) | (arr_rt.sum(axis=2) > 10)
    return 100.0 * float(dark_mask.sum()) / float(visible.sum())


def _non_shadow_max_diff(arr_no: np.ndarray, arr_rt: np.ndarray) -> float:
    """Max per-channel diff among pixels that did NOT get darker."""
    diff = np.abs(arr_rt.astype(np.float32) - arr_no.astype(np.float32))
    dark_mask = (arr_no.astype(np.float32) - arr_rt.astype(np.float32)).mean(axis=2) > DARK_THRESHOLD
    non_dark = ~dark_mask
    if not non_dark.any():
        return 0.0
    return float(diff[non_dark].max())


def _make_comparison(arr_no, arr_rt, out_path: Path):
    """Save a side-by-side comparison image."""
    H = arr_no.shape[0]
    gap = 4
    W = arr_no.shape[1] + gap + arr_rt.shape[1]
    canvas = np.full((H, W, 3), 255, dtype=np.uint8)
    canvas[:, :arr_no.shape[1]] = arr_no
    canvas[:, arr_no.shape[1] + gap:] = arr_rt
    Image.fromarray(canvas, "RGB").save(out_path)


# --- Genesis test ----------------------------------------------------------
def run_genesis_test() -> dict:
    from rtxns_genesis_style import CameraDesc, GenesisStyleRenderer, SurfaceDesc

    print("\n=== Genesis shadow regression ===")

    def _make_box(sx, sy, sz):
        s = (0.5 * sx, 0.5 * sy, 0.5 * sz)
        faces = [
            [(-s[0], -s[1], -s[2]), (s[0], -s[1], -s[2]), (s[0], s[1], -s[2]), (-s[0], s[1], -s[2])],
            [(-s[0], -s[1], s[2]), (-s[0], s[1], s[2]), (s[0], s[1], s[2]), (s[0], -s[1], s[2])],
            [(-s[0], -s[1], -s[2]), (-s[0], -s[1], s[2]), (s[0], -s[1], s[2]), (s[0], -s[1], -s[2])],
            [(s[0], -s[1], -s[2]), (s[0], -s[1], s[2]), (s[0], s[1], s[2]), (s[0], s[1], -s[2])],
            [(-s[0], s[1], -s[2]), (s[0], s[1], -s[2]), (s[0], s[1], s[2]), (-s[0], s[1], s[2])],
            [(-s[0], -s[1], -s[2]), (-s[0], s[1], -s[2]), (-s[0], s[1], s[2]), (-s[0], -s[1], s[2])],
        ]
        face_normals = [(0, 0, -1), (0, 0, 1), (0, -1, 0), (1, 0, 0), (0, 1, 0), (-1, 0, 0)]
        verts = np.array([v for f in faces for v in f], dtype=np.float32)
        verts[:, 1] += s[1]
        norms = np.array([n for n in face_normals for _ in range(4)], dtype=np.float32)
        tri_base = np.array([[0, 2, 1], [0, 3, 2]], dtype=np.uint32)
        tris = np.vstack([tri_base + i * 4 for i in range(6)])
        return verts, tris, norms

    def _make_plane(size):
        h = size * 0.5
        verts = np.array([(-h, 0, -h), (h, 0, -h), (h, 0, h), (-h, 0, h)], dtype=np.float32)
        tris = np.array([(0, 2, 1), (0, 3, 2)], dtype=np.uint32)
        return verts, tris

    with GenesisStyleRenderer(module_dir=repo_root / "bin" / "windows-x64", runtime_dir=repo_root) as r:
        r.set_ambient((0.15, 0.14, 0.13), (0.10, 0.09, 0.08))
        r.set_default_light(direction=(-0.7, -1.0, -0.3), color=(1.0, 0.95, 0.88), irradiance=1.2)
        r.add_surface("ground", SurfaceDesc(base_color=(0.7, 0.7, 0.8, 1.0), roughness=0.9))
        r.add_rigid("ground", *_make_plane(8.0))
        r.add_surface("box", SurfaceDesc(base_color=(0.91, 0.50, 0.18, 1.0), roughness=0.5))
        r.add_rigid("box", *_make_box(1.0, 1.0, 1.0))
        cam = CameraDesc(uid="main", pos=(3.0, 2.5, 4.0), lookat=(0, 0.3, 0), up=(0, 1, 0), res=(640, 480), fov=50)
        r.add_camera(cam)

        r._scene.enable_rt_shadows(False)
        img_no = r.render_camera(cam, force_render=True)
        stats_no = r._scene.get_last_frame_stats()

        r._scene.enable_rt_shadows(True)
        img_rt = r.render_camera(cam, force_render=True)
        stats_rt = r._scene.get_last_frame_stats()

    arr_no = np.array(img_no, dtype=np.uint8)[:, :, :3]
    arr_rt = np.array(img_rt, dtype=np.uint8)[:, :, :3]

    Image.fromarray(arr_no, "RGB").save(REPORT_DIR / "genesis_no_shadow.png")
    Image.fromarray(arr_rt, "RGB").save(REPORT_DIR / "genesis_rt_shadow.png")
    _make_comparison(arr_no, arr_rt, REPORT_DIR / "genesis_comparison.png")

    ratio = _dark_ratio(arr_no, arr_rt)
    ns_diff = _non_shadow_max_diff(arr_no, arr_rt)
    print(f"  dark ratio: {ratio:.1f}%  (expected {GENESIS_EXPECTED[0]:.1f}-{GENESIS_EXPECTED[1]:.1f}%)")
    print(f"  non-shadow max diff: {ns_diff:.1f}  (limit {NON_SHADOW_MAX_DIFF})")
    print(f"  stats[no]:  total={stats_no['total_ms']:.1f}ms  raster={stats_no['raster_ms']:.1f}ms")
    print(f"  stats[rt]:  total={stats_rt['total_ms']:.1f}ms  raster={stats_rt['raster_ms']:.1f}ms  "
          f"blas={stats_rt['blas_build_ms']:.1f}ms  shadow_ray={stats_rt['shadow_ray_ms']:.1f}ms  "
          f"composite={stats_rt['composite_ms']:.1f}ms")

    ok = True
    if not (GENESIS_EXPECTED[0] <= ratio <= GENESIS_EXPECTED[1]):
        print(f"  FAIL: dark ratio {ratio:.1f}% out of range")
        ok = False
    if ns_diff > NON_SHADOW_MAX_DIFF:
        print(f"  FAIL: non-shadow max diff {ns_diff:.1f} exceeds {NON_SHADOW_MAX_DIFF}")
        ok = False
    if ok:
        print("  PASS")
    return {
        "scene": "genesis",
        "pass": ok,
        "dark_ratio": ratio,
        "non_shadow_max_diff": ns_diff,
        "stats_no_shadow": stats_no,
        "stats_rt_shadow": stats_rt,
    }


# --- Bistro test -----------------------------------------------------------
def run_bistro_test() -> dict:
    import ctypes
    import math
    import struct
    import DonutRenderPyNative as rr

    print("\n=== Bistro shadow regression ===")

    class _NiagaraCamera(ctypes.Structure):
        _fields_ = [
            ("position", ctypes.c_float * 3),
            ("orientation", ctypes.c_float * 4),
            ("fov_y", ctypes.c_float),
            ("znear", ctypes.c_float),
        ]

    class _NiagaraSceneHeader(ctypes.Structure):
        _fields_ = [
            ("magic", ctypes.c_uint32), ("version", ctypes.c_uint32),
            ("meshlet_max_vertices", ctypes.c_uint32), ("meshlet_max_triangles", ctypes.c_uint32),
            ("clrt_mode", ctypes.c_bool), ("compressed", ctypes.c_bool),
            ("compressed_vertex_bytes", ctypes.c_uint32), ("compressed_index_bytes", ctypes.c_uint32),
            ("compressed_meshlet_data_bytes", ctypes.c_uint32), ("compressed_meshlet_vtx0_bytes", ctypes.c_uint32),
            ("vertex_count", ctypes.c_uint32), ("index_count", ctypes.c_uint32),
            ("meshlet_count", ctypes.c_uint32), ("meshletdata_count", ctypes.c_uint32),
            ("meshletvtx0_count", ctypes.c_uint32), ("mesh_count", ctypes.c_uint32),
            ("material_count", ctypes.c_uint32), ("draw_count", ctypes.c_uint32),
            ("texture_path_count", ctypes.c_uint32), ("omm_array_data_size", ctypes.c_uint32),
            ("omm_index_data_size", ctypes.c_uint32), ("omm_desc_count", ctypes.c_uint32),
            ("omm_states", ctypes.c_uint32), ("camera", _NiagaraCamera),
            ("sun_direction", ctypes.c_float * 3),
        ]

    def _rotate_by_quat_xyzw(q, v):
        x, y, z, w = q
        cx = y * v[2] - z * v[1]; cy = z * v[0] - x * v[2]; cz = x * v[1] - y * v[0]
        tx, ty, tz = 2.0 * cx, 2.0 * cy, 2.0 * cz
        c2x = y * tz - z * ty; c2y = z * tx - x * tz; c2z = x * ty - y * tx
        return [v[0] + w * tx + c2x, v[1] + w * ty + c2y, v[2] + w * tz + c2z]

    scene_dir = Path(r"D:\niagara_bistro")
    header_blob = (scene_dir / "bistro.gltf.cache").read_bytes()[: ctypes.sizeof(_NiagaraSceneHeader)]
    header = _NiagaraSceneHeader.from_buffer_copy(header_blob)
    camera_blob = (scene_dir / "bistro.gltf.camera").read_bytes()
    _, px, py, pz, qx, qy, qz, qw = struct.unpack("<I3f4f", camera_blob)
    position = [px, py, pz]; quat = [qx, qy, qz, qw]
    forward = _rotate_by_quat_xyzw(quat, [0.0, 0.0, -1.0])
    up = _rotate_by_quat_xyzw(quat, [0.0, 1.0, 0.0])
    target = [position[i] + forward[i] for i in range(3)]
    fov_degrees = float(header.camera.fov_y) * 180.0 / math.pi
    z_near = float(header.camera.znear)
    sun_direction = [float(v) for v in header.sun_direction]

    rr.init(runtime_dir=str(repo_root), backend="vulkan", device_index=-1, enable_debug=False)
    scene = rr.create_scene()
    width, height = 1024, 768

    print("  Loading Bistro...")
    t0 = time.time()
    scene.load_scene(str(scene_dir / "bistro.gltf"))
    print(f"  Loaded in {time.time() - t0:.1f}s")

    scene.set_camera(position=position, target=target, up=up, fov_degrees=fov_degrees,
                     width=width, height=height, z_near=z_near, z_far=200.0)
    scene.set_default_light(direction=[-v for v in sun_direction], color=[1.0, 1.0, 1.0], irradiance=2.0)
    scene.set_ambient(top_rgb=[0.12, 0.12, 0.12], bottom_rgb=[0.06, 0.06, 0.06])

    scene.enable_rt_shadows(False)
    img_no = scene.render_frame()
    stats_no = scene.get_last_frame_stats()

    scene.enable_rt_shadows(True)
    img_rt = scene.render_frame()
    stats_rt = scene.get_last_frame_stats()

    rr.destroy()

    arr_no = np.frombuffer(img_no, np.uint8).reshape(height, width, 4)[:, :, :3].copy()
    arr_rt = np.frombuffer(img_rt, np.uint8).reshape(height, width, 4)[:, :, :3].copy()

    Image.fromarray(arr_no, "RGB").save(REPORT_DIR / "bistro_no_shadow.png")
    Image.fromarray(arr_rt, "RGB").save(REPORT_DIR / "bistro_rt_shadow.png")
    _make_comparison(arr_no, arr_rt, REPORT_DIR / "bistro_comparison.png")

    ratio = _dark_ratio(arr_no, arr_rt)
    ns_diff = _non_shadow_max_diff(arr_no, arr_rt)
    print(f"  dark ratio: {ratio:.1f}%  (expected {BISTRO_EXPECTED[0]:.1f}-{BISTRO_EXPECTED[1]:.1f}%)")
    print(f"  non-shadow max diff: {ns_diff:.1f}  (limit {NON_SHADOW_MAX_DIFF})")
    print(f"  stats[no]:  total={stats_no['total_ms']:.1f}ms  raster={stats_no['raster_ms']:.1f}ms")
    print(f"  stats[rt]:  total={stats_rt['total_ms']:.1f}ms  raster={stats_rt['raster_ms']:.1f}ms  "
          f"blas={stats_rt['blas_build_ms']:.1f}ms  shadow_ray={stats_rt['shadow_ray_ms']:.1f}ms  "
          f"composite={stats_rt['composite_ms']:.1f}ms")

    ok = True
    if not (BISTRO_EXPECTED[0] <= ratio <= BISTRO_EXPECTED[1]):
        print(f"  FAIL: dark ratio {ratio:.1f}% out of range")
        ok = False
    if ns_diff > NON_SHADOW_MAX_DIFF:
        print(f"  FAIL: non-shadow max diff {ns_diff:.1f} exceeds {NON_SHADOW_MAX_DIFF}")
        ok = False
    if ok:
        print("  PASS")
    return {
        "scene": "bistro",
        "pass": ok,
        "dark_ratio": ratio,
        "non_shadow_max_diff": ns_diff,
        "stats_no_shadow": stats_no,
        "stats_rt_shadow": stats_rt,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-bistro", action="store_true")
    args = parser.parse_args()

    results = []
    results.append(run_genesis_test())
    if not args.skip_bistro:
        results.append(run_bistro_test())

    report_path = REPORT_DIR / "regression_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nReport saved to {report_path}")

    all_pass = all(r["pass"] for r in results)
    print("\n=== Summary ===")
    for r in results:
        status = "PASS" if r["pass"] else "FAIL"
        print(f"  {r['scene']:10s}: {status}  (dark={r['dark_ratio']:.1f}%)")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
