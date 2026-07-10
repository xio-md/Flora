"""Capture blur-off / blur-on shadow images + frame stats for report."""
import ctypes, math, struct, sys, time
from pathlib import Path

import numpy as np
from PIL import Image

repo_root = Path(r"D:\RTXNS")
sys.path.insert(0, str(repo_root / "bin" / "windows-x64"))
import DonutRenderPyNative as rr


def _rotate_by_quat_xyzw(q, v):
    x, y, z, w = q
    cx = y * v[2] - z * v[1]
    cy = z * v[0] - x * v[2]
    cz = x * v[1] - y * v[0]
    tx, ty, tz = 2.0 * cx, 2.0 * cy, 2.0 * cz
    c2x = y * tz - z * ty
    c2y = z * tx - x * tz
    c2z = x * ty - y * tx
    return [v[0] + w * tx + c2x, v[1] + w * ty + c2y, v[2] + w * tz + c2z]


class _NiagaraCamera(ctypes.Structure):
    _fields_ = [
        ("position", ctypes.c_float * 3),
        ("orientation", ctypes.c_float * 4),
        ("fov_y", ctypes.c_float),
        ("znear", ctypes.c_float),
    ]


class _NiagaraSceneHeader(ctypes.Structure):
    _fields_ = [
        ("magic", ctypes.c_uint32),
        ("version", ctypes.c_uint32),
        ("meshlet_max_vertices", ctypes.c_uint32),
        ("meshlet_max_triangles", ctypes.c_uint32),
        ("clrt_mode", ctypes.c_bool),
        ("compressed", ctypes.c_bool),
        ("compressed_vertex_bytes", ctypes.c_uint32),
        ("compressed_index_bytes", ctypes.c_uint32),
        ("compressed_meshlet_data_bytes", ctypes.c_uint32),
        ("compressed_meshlet_vtx0_bytes", ctypes.c_uint32),
        ("vertex_count", ctypes.c_uint32),
        ("index_count", ctypes.c_uint32),
        ("meshlet_count", ctypes.c_uint32),
        ("meshletdata_count", ctypes.c_uint32),
        ("meshletvtx0_count", ctypes.c_uint32),
        ("mesh_count", ctypes.c_uint32),
        ("material_count", ctypes.c_uint32),
        ("draw_count", ctypes.c_uint32),
        ("texture_path_count", ctypes.c_uint32),
        ("omm_array_data_size", ctypes.c_uint32),
        ("omm_index_data_size", ctypes.c_uint32),
        ("omm_desc_count", ctypes.c_uint32),
        ("omm_states", ctypes.c_uint32),
        ("camera", _NiagaraCamera),
        ("sun_direction", ctypes.c_float * 3),
    ]


def load_niagara_inputs(scene_dir):
    scene_dir = Path(scene_dir)
    header_blob = (scene_dir / "bistro.gltf.cache").read_bytes()[: ctypes.sizeof(_NiagaraSceneHeader)]
    header = _NiagaraSceneHeader.from_buffer_copy(header_blob)
    if header.magic != 0x434E4353 or header.version != 5:
        raise RuntimeError("Unexpected Niagara scene cache header")
    camera_blob = (scene_dir / "bistro.gltf.camera").read_bytes()
    camera_version, px, py, pz, qx, qy, qz, qw = struct.unpack("<I3f4f", camera_blob)
    if camera_version != 1:
        raise RuntimeError("Unexpected Niagara camera file version")
    position = [px, py, pz]
    quat = [qx, qy, qz, qw]
    forward = _rotate_by_quat_xyzw(quat, [0.0, 0.0, -1.0])
    up = _rotate_by_quat_xyzw(quat, [0.0, 1.0, 0.0])
    target = [position[i] + forward[i] for i in range(3)]
    fov_degrees = float(header.camera.fov_y) * 180.0 / math.pi
    z_near = float(header.camera.znear)
    sun_direction = [float(v) for v in header.sun_direction]
    return position, target, up, fov_degrees, z_near, sun_direction


def render_and_save(scene, label, filename, stats_list):
    img = scene.render_frame()
    arr = np.frombuffer(img, np.uint8).reshape(768, 1024, 4)[:, :, :3]
    Image.fromarray(arr, "RGB").save(filename)

    stats = scene.get_last_frame_stats()
    stats_list.append((label, dict(stats)))
    fps = 1000.0 / stats["total_ms"] if stats["total_ms"] > 0 else 0
    print(f"  {label}: mean={arr.mean():.1f}")
    print(f"    total={stats['total_ms']:.1f}ms  fps={fps:.0f}"
          f"  raster={stats['raster_ms']:.1f}ms"
          f"  blas={stats['blas_build_ms']:.1f}ms"
          f"  shadow_ray={stats['shadow_ray_ms']:.1f}ms"
          f"  composite={stats['composite_ms']:.1f}ms")


def main():
    rr.init(runtime_dir=str(repo_root), backend="vulkan", device_index=-1, enable_debug=False)
    scene = rr.create_scene()
    scene_dir = Path(r"D:\niagara_bistro")
    position, target, up, fov_degrees, z_near, sun_direction = load_niagara_inputs(scene_dir)
    width, height = 1024, 768

    print("Loading Bistro...")
    t0 = time.time()
    scene.load_scene(str(scene_dir / "bistro.gltf"))
    print(f"  Loaded in {time.time() - t0:.1f}s")

    scene.set_camera(position=position, target=target, up=up, fov_degrees=fov_degrees,
                     width=width, height=height, z_near=z_near, z_far=200.0)
    scene.set_default_light(direction=[-v for v in sun_direction],
                            color=[1.0, 1.0, 1.0], irradiance=50.0)
    scene.set_ambient(top_rgb=[0.03, 0.03, 0.03], bottom_rgb=[0.01, 0.01, 0.01])

    out = Path(r"D:\RTXNS\output\bistro_test")
    out.mkdir(parents=True, exist_ok=True)
    stats_list = []

    # --- 1. No shadow (composite-only baseline) ---
    scene.enable_rt_shadows(False)
    render_and_save(scene, "no_shadow", out / "bistro_no_shadow.png", stats_list)

    # --- 2. RT shadow, blur OFF (sun jitter only) ---
    scene.enable_rt_shadows(True)
    scene.enable_shadow_blur(False)
    render_and_save(scene, "sun_jitter_NO_blur", out / "bistro_sun_jitter_no_blur.png", stats_list)

    # --- 3. RT shadow, blur ON (full optimization) ---
    scene.enable_shadow_blur(True)
    render_and_save(scene, "sun_jitter_WITH_blur", out / "bistro_rt_shadow.png", stats_list)

    # --- 4. Second frame with blur ON (steady state, no BLAS build) ---
    render_and_save(scene, "steady_blur_ON", out / "bistro_rt_shadow_steady.png", stats_list)

    # --- Summary ---
    print("\n=== Frame Stats Summary ===")
    for label, s in stats_list:
        fps = 1000.0 / s["total_ms"] if s["total_ms"] > 0 else 0
        print(f"  {label}: {s['total_ms']:.1f}ms ({fps:.0f} FPS)"
              f"  shadow_ray={s['shadow_ray_ms']:.1f}ms  composite={s['composite_ms']:.1f}ms"
              f"  blas={s['blas_build_ms']:.1f}ms")

    rr.destroy()
    print("Done.")


if __name__ == "__main__":
    main()
