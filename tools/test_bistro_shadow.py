"""Bistro RT shadow - Niagara camera/cache inputs."""
import ctypes
import math
import struct
import sys
import time
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
    print(f"Niagara camera: pos={position}, target={target}, up={up}, fov={fov_degrees:.2f}")
    print(f"Niagara sunDirection: {sun_direction}")

    scene.set_camera(
        position=position,
        target=target,
        up=up,
        fov_degrees=fov_degrees,
        width=width,
        height=height,
        z_near=z_near,
        z_far=200.0,
    )
    scene.set_default_light(
        direction=[-v for v in sun_direction],
        color=[1.0, 1.0, 1.0],
        irradiance=2.0,
    )
    scene.set_ambient(top_rgb=[0.12, 0.12, 0.12], bottom_rgb=[0.06, 0.06, 0.06])

    out = Path(r"D:\RTXNS\output\bistro_test")
    out.mkdir(parents=True, exist_ok=True)

    scene.enable_rt_shadows(False)
    img_no = scene.render_frame()
    arr_no = np.frombuffer(img_no, np.uint8).reshape(height, width, 4)[:, :, :3]
    Image.fromarray(arr_no, "RGB").save(out / "bistro_no_shadow.png")
    print(
        "No-shadow: "
        f"mean={arr_no.mean():.1f}, "
        f"isGrey={(arr_no[:, :, 0] == arr_no[:, :, 1]).all() and (arr_no[:, :, 1] == arr_no[:, :, 2]).all()}"
    )

    scene.enable_rt_shadows(True)
    img_rt = scene.render_frame()
    arr_rt = np.frombuffer(img_rt, np.uint8).reshape(height, width, 4)[:, :, :3]
    Image.fromarray(arr_rt, "RGB").save(out / "bistro_rt_shadow.png")
    print(
        "RT-shadow: "
        f"mean={arr_rt.mean():.1f}, "
        f"isGrey={(arr_rt[:, :, 0] == arr_rt[:, :, 1]).all() and (arr_rt[:, :, 1] == arr_rt[:, :, 2]).all()}"
    )
    diff = (np.abs(arr_rt.astype(float) - arr_no.astype(float)).mean(axis=2) > 10).sum()
    print(f"Pixels darkened: {diff}/{width * height} ({100 * diff / (width * height):.1f}%)")

    rr.destroy()
    print("Done.")


if __name__ == "__main__":
    main()
