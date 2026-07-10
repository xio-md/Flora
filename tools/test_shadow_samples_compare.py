"""Multi-sample shadow quality comparison: 4 vs 8 vs 16 samples.

Renders Bistro with different shadow sample counts and composes a side-by-side
comparison image for visual quality assessment.
"""
import ctypes, math, struct, sys, time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

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


class _NH(ctypes.Structure):
    _fields_ = [
        ("mag", ctypes.c_uint32), ("ver", ctypes.c_uint32),
        ("mmv", ctypes.c_uint32), ("mmt", ctypes.c_uint32),
        ("clrt", ctypes.c_bool), ("cmp", ctypes.c_bool),
        ("cvb", ctypes.c_uint32), ("cib", ctypes.c_uint32),
        ("cmd", ctypes.c_uint32), ("cmv", ctypes.c_uint32),
        ("vc", ctypes.c_uint32), ("ic", ctypes.c_uint32),
        ("mc", ctypes.c_uint32), ("mdc", ctypes.c_uint32),
        ("mvc", ctypes.c_uint32), ("meshc", ctypes.c_uint32),
        ("matc", ctypes.c_uint32), ("dc", ctypes.c_uint32),
        ("tpc", ctypes.c_uint32), ("oads", ctypes.c_uint32),
        ("oids", ctypes.c_uint32), ("odc", ctypes.c_uint32),
        ("oms", ctypes.c_uint32),
        ("cam_p", ctypes.c_float * 3),
        ("cam_o", ctypes.c_float * 4),
        ("cam_f", ctypes.c_float),
        ("cam_z", ctypes.c_float),
        ("sd", ctypes.c_float * 3),
    ]


def load_niagara_inputs(scene_dir):
    scene_dir = Path(scene_dir)
    hb = (scene_dir / "bistro.gltf.cache").read_bytes()[:ctypes.sizeof(_NH)]
    h = _NH.from_buffer_copy(hb)
    cb = (scene_dir / "bistro.gltf.camera").read_bytes()
    cv, px, py, pz, qx, qy, qz, qw = struct.unpack("<I3f4f", cb)
    pos = [px, py, pz]
    quat = [qx, qy, qz, qw]
    fw = _rotate_by_quat_xyzw(quat, [0, 0, -1])
    up = _rotate_by_quat_xyzw(quat, [0, 1, 0])
    tgt = [pos[i] + fw[i] for i in range(3)]
    fov = float(h.cam_f) * 180 / math.pi
    zn = float(h.cam_z)
    sun = [float(v) for v in h.sd]
    return pos, tgt, up, fov, zn, sun


def render_with_samples(samples, out_dir, label):
    """Render bistro with given shadow sample count, return (arr, stats)."""
    rr.init(runtime_dir=str(repo_root), backend="vulkan", device_index=-1, enable_debug=False)
    scene = rr.create_scene()
    sd = Path(r"D:\niagara_bistro")
    pos, tgt, up, fov, zn, sun = load_niagara_inputs(sd)
    W, H = 1024, 768

    scene.load_scene(str(sd / "bistro.gltf"))
    scene.set_camera(position=pos, target=tgt, up=up, fov_degrees=fov,
                     width=W, height=H, z_near=zn, z_far=200.0)
    scene.set_default_light(direction=[-v for v in sun], color=[1, 1, 1], irradiance=50)
    scene.set_ambient(top_rgb=[0.03, 0.03, 0.03], bottom_rgb=[0.01, 0.01, 0.01])
    scene.enable_rt_shadows(True)
    scene.enable_shadow_blur(True)
    scene.set_shadow_samples(samples)

    # Warm up (BLAS build)
    scene.render_frame()
    # Capture steady state
    img = scene.render_frame()
    st = scene.get_last_frame_stats()
    arr = np.frombuffer(img, np.uint8).reshape(H, W, 4)[:, :, :3]
    fname = out_dir / f"shadow_s{samples}.png"
    Image.fromarray(arr, "RGB").save(fname)
    print(f"  {label}: mean={arr.mean():.1f} ray={st['shadow_ray_ms']:.2f}ms total={st['total_ms']:.1f}ms")
    rr.destroy()
    return arr, st, fname


def main():
    out = Path(r"D:\RTXNS\output\bistro_test")
    out.mkdir(parents=True, exist_ok=True)

    results = []
    for n, label in [(4, "4 samples"), (8, "8 samples"), (16, "16 samples")]:
        print(f"Rendering {label}...")
        arr, st, fname = render_with_samples(n, out, label)
        results.append((n, label, arr, st, fname))

    # Compose side-by-side comparison
    W, H = 1024, 768
    PAD = 20
    LABEL_H = 50
    total_w = W * 3 + PAD * 4
    total_h = H + LABEL_H + PAD * 2

    canvas = Image.new("RGB", (total_w, total_h), (16, 16, 16))
    d = ImageDraw.Draw(canvas)
    try:
        f = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", 22)
        fs = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", 16)
    except OSError:
        f = ImageFont.load_default()
        fs = f

    for i, (n, label, arr, st, _) in enumerate(results):
        x = PAD + i * (W + PAD)
        canvas.paste(Image.fromarray(arr, "RGB"), (x, LABEL_H))
        d.text((x + 8, PAD // 2), label, font=f, fill=(255, 255, 255))
        d.text((x + 8, PAD // 2 + 26),
               f"ray={st['shadow_ray_ms']:.2f}ms  total={st['total_ms']:.1f}ms",
               font=fs, fill=(200, 200, 200))

    poster = out / "multisample_shadow_compare.png"
    canvas.save(poster)
    print(f"\nWrote {poster} ({canvas.size[0]}x{canvas.size[1]})")

    # Also save pixel-diff images
    for i in range(1, len(results)):
        prev_arr = results[i - 1][2].astype(float)
        curr_arr = results[i][2].astype(float)
        diff = np.abs(curr_arr - prev_arr).max(axis=2)
        diff_vis = np.clip(diff * 5, 0, 255).astype(np.uint8)
        diff_fname = out / f"multisample_diff_s{results[i-1][0]}_s{results[i][0]}.png"
        Image.fromarray(diff_vis, "L").save(diff_fname)
        print(f"  diff {results[i-1][0]}->{results[i][0]}: mean={diff.mean():.2f} max={diff.max():.0f}")


if __name__ == "__main__":
    main()
