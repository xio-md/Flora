"""Quick RT shadow test — ground plane + box to verify directional shadow."""
import sys, numpy as np
from pathlib import Path

repo_root = Path(r"D:\RTXNS")
sys.path.insert(0, str(repo_root / "bin" / "windows-x64"))
sys.path.insert(0, str(repo_root / "python"))
from rtxns_genesis_style import CameraDesc, GenesisStyleRenderer, SurfaceDesc

def _make_box(sx, sy, sz):
    s = (0.5*sx, 0.5*sy, 0.5*sz)
    faces = [
        [(-s[0],-s[1],-s[2]),( s[0],-s[1],-s[2]),( s[0], s[1],-s[2]),(-s[0], s[1],-s[2])],
        [(-s[0],-s[1], s[2]),(-s[0], s[1], s[2]),( s[0], s[1], s[2]),( s[0],-s[1], s[2])],
        [(-s[0],-s[1],-s[2]),(-s[0],-s[1], s[2]),( s[0],-s[1], s[2]),( s[0],-s[1],-s[2])],
        [( s[0],-s[1],-s[2]),( s[0],-s[1], s[2]),( s[0], s[1], s[2]),( s[0], s[1],-s[2])],
        [(-s[0], s[1],-s[2]),( s[0], s[1],-s[2]),( s[0], s[1], s[2]),(-s[0], s[1], s[2])],
        [(-s[0],-s[1],-s[2]),(-s[0], s[1],-s[2]),(-s[0], s[1], s[2]),(-s[0],-s[1], s[2])],
    ]
    face_normals = [
        (0, 0, -1),  # Z-: facing -Z
        (0, 0,  1),  # Z+: facing +Z
        (0, -1, 0),  # Y-: bottom
        (1, 0,  0),  # X+: facing +X
        (0, 1,  0),  # Y+: top
        (-1, 0, 0),  # X-: facing -X
    ]
    verts = np.array([v for f in faces for v in f], dtype=np.float32)
    verts[:, 1] += s[1]  # Put the box on the ground: y = 0..sy.
    norms = np.array([n for n in face_normals for _ in range(4)], dtype=np.float32)
    tri_base = np.array([[0,2,1],[0,3,2]], dtype=np.uint32)
    tris = np.vstack([tri_base + i*4 for i in range(6)])
    return verts, tris, norms

def _make_plane(size):
    h = size * 0.5
    verts = np.array([(-h,0,-h),(h,0,-h),(h,0,h),(-h,0,h)], dtype=np.float32)
    tris = np.array([(0,2,1),(0,3,2)], dtype=np.uint32)
    return verts, tris

def main():
    with GenesisStyleRenderer(module_dir=repo_root / "bin" / "windows-x64", runtime_dir=repo_root) as r:
        r.set_ambient((0.15,0.14,0.13),(0.10,0.09,0.08))
        r.set_default_light(direction=(-0.7,-1.0,-0.3), color=(1.0,0.95,0.88), irradiance=1.2)
        r.add_surface("ground", SurfaceDesc(base_color=(0.7,0.7,0.8,1.0), roughness=0.9))
        r.add_rigid("ground", *_make_plane(8.0))
        r.add_surface("box", SurfaceDesc(base_color=(0.91,0.50,0.18,1.0), roughness=0.5))
        r.add_rigid("box", *_make_box(1.0, 1.0, 1.0))
        cam = CameraDesc(uid="main", pos=(3.0, 2.5, 4.0), lookat=(0, 0.3, 0), up=(0,1,0), res=(640,480), fov=50)
        r.add_camera(cam)

        out_dir = Path(r"D:\RTXNS\output\shadow_test")
        out_dir.mkdir(parents=True, exist_ok=True)

        r._scene.enable_rt_shadows(False)
        img_no = r.render_camera(cam, force_render=True)
        _save(out_dir / "no_shadow.ppm", img_no)
        print(f"No-shadow: mean={img_no.mean():.1f}")

        r._scene.enable_rt_shadows(True)
        img_rt = r.render_camera(cam, force_render=True)
        _save(out_dir / "rt_shadow.ppm", img_rt)
        print(f"RT-shadow: mean={img_rt.mean():.1f}")

        diff = np.abs(img_rt.astype(np.float32) - img_no.astype(np.float32))
        print(f"Max diff: {diff.max():.0f}, Mean diff: {diff.mean():.1f}")

    print("Done.")

def _save(path, img):
    h, w, c = img.shape
    with open(path, 'wb') as f:
        f.write(f"P6\n{w} {h}\n255\n".encode())
        f.write(img[:,:,:3].astype(np.uint8).tobytes())

if __name__ == "__main__":
    main()
