"""OMM comprehensive experiment — multi-config data collection."""
import ctypes, math, struct, sys, time, numpy as np
from pathlib import Path
from PIL import Image

repo_root = Path(r"D:\RTXNS")
sys.path.insert(0, str(repo_root / "bin" / "windows-x64"))
import DonutRenderPyNative as rr

def _r(q,v):
    x,y,z,w=q; cx=y*v[2]-z*v[1]; cy=z*v[0]-x*v[2]; cz=x*v[1]-y*v[0]
    tx,ty,tz=2*cx,2*cy,2*cz
    return [v[0]+w*tx+y*tz-z*ty, v[1]+w*ty+z*tx-x*tz, v[2]+w*tz+x*ty-y*tx]
_NC=type('',(ctypes.Structure,),{'_fields_':[("p",ctypes.c_float*3),("o",ctypes.c_float*4),("f",ctypes.c_float),("z",ctypes.c_float)]})
_NH=type('',(ctypes.Structure,),{'_fields_':
    [("mag",ctypes.c_uint32),("ver",ctypes.c_uint32),("mmv",ctypes.c_uint32),("mmt",ctypes.c_uint32),
     ("clrt",ctypes.c_bool),("cmp",ctypes.c_bool),("cvb",ctypes.c_uint32),("cib",ctypes.c_uint32),
     ("cmd",ctypes.c_uint32),("cmv",ctypes.c_uint32),("vc",ctypes.c_uint32),("ic",ctypes.c_uint32),
     ("mc",ctypes.c_uint32),("mdc",ctypes.c_uint32),("mvc",ctypes.c_uint32),("meshc",ctypes.c_uint32),
     ("matc",ctypes.c_uint32),("dc",ctypes.c_uint32),("tpc",ctypes.c_uint32),
     ("oads",ctypes.c_uint32),("oids",ctypes.c_uint32),("odc",ctypes.c_uint32),("oms",ctypes.c_uint32),
     ("cam",_NC),("sd",ctypes.c_float*3)]})
sd=Path(r"D:\niagara_bistro")
hb=(sd/"bistro.gltf.cache").read_bytes()[:ctypes.sizeof(_NH)]
h=_NH.from_buffer_copy(hb)
cv,px,py,pz,qx,qy,qz,qw=struct.unpack("<I3f4f",(sd/"bistro.gltf.camera").read_bytes())
pos=[px,py,pz]; fw=_r([qx,qy,qz,qw],[0,0,-1]); up=_r([qx,qy,qz,qw],[0,1,0])
tgt=[pos[i]+fw[i] for i in range(3)]
fov=float(h.cam.f)*180/math.pi; zn=float(h.cam.z)
sun=[float(v) for v in h.sd]

out = Path(r"D:\RTXNS\output\bistro_test")
W, H = 1024, 768

def render_config(label, omm=False, stress=False, subdiv=5, fmt=2, samples=4):
    rr.init(runtime_dir=str(repo_root), backend="vulkan", device_index=-1, enable_debug=False)
    s = rr.create_scene()
    s.load_scene(str(sd / "bistro.gltf"))
    s.set_camera(position=pos, target=tgt, up=up, fov_degrees=fov, width=W, height=H, z_near=zn, z_far=200)
    s.set_default_light(direction=[-v for v in sun], color=[1,1,1], irradiance=50)
    s.set_ambient(top_rgb=[0.03]*3, bottom_rgb=[0.01]*3)
    s.enable_rt_shadows(True)
    s.enable_omm(omm)
    s.enable_omm_stress(stress)
    s.set_omm_config(subdiv, fmt)
    s.set_shadow_samples(samples)

    # First frame
    img = s.render_frame()
    f0 = s.get_last_frame_stats()

    # Save image
    arr = np.frombuffer(img, np.uint8).reshape(H, W, 4)[:, :, :3]
    Image.fromarray(arr, "RGB").save(out / f"exp_{label}.png")

    # Steady frames
    rays = []
    for i in range(3):
        s.render_frame()
        st = s.get_last_frame_stats()
        rays.append(st["shadow_ray_ms"])

    avg_ray = sum(rays) / len(rays)
    rr.destroy()

    return {
        "label": label,
        "blas_ms": f0["blas_build_ms"],
        "total_ms": f0["total_ms"],
        "shadow_ray_steady_ms": avg_ray,
        "img_mean": arr.mean(),
        "fs": f0
    }

results = []

# === Experiment 1: OMM ON vs OFF correctness ===
print("=== Exp1: OMM ON vs OFF ===")
r = render_config("no_omm", omm=False); results.append(r)
print(f"  OMM OFF: blas={r['blas_ms']:.0f}ms  ray={r['shadow_ray_steady_ms']:.2f}ms")
r = render_config("omm_4s", omm=True, subdiv=5, fmt=2); results.append(r)
print(f"  OMM ON:  blas={r['blas_ms']:.0f}ms  ray={r['shadow_ray_steady_ms']:.2f}ms")

# === Experiment 2: OMM OFF/ON pixel diff ===
off = np.array(Image.open(out/"exp_no_omm.png"))[:,:,:3].astype(float)
on  = np.array(Image.open(out/"exp_omm_4s.png"))[:,:,:3].astype(float)
diff = np.abs(on - off).max(axis=2)
diff_vis = np.clip(diff * 20, 0, 255).astype(np.uint8)
Image.fromarray(diff_vis, "L").save(out / "exp_omm_diff.png")
print(f"\n  Pixel diff: max={diff.max():.0f}  mean={diff.mean():.2f}")

# === Experiment 3: Subdivision level sweep ===
print("\n=== Exp3: Subdivision level sweep ===")
for lv in [3, 4, 5]:
    r = render_config(f"subdiv_{lv}", omm=True, subdiv=lv)
    results.append(r)
    print(f"  Lv{lv}: blas={r['blas_ms']:.0f}ms  ray={r['shadow_ray_steady_ms']:.2f}ms")

# === Experiment 4: 2-state vs 4-state ===
print("\n=== Exp4: Format comparison ===")
r = render_config("omm_2s", omm=True, subdiv=5, fmt=1); results.append(r)
print(f"  2-state: blas={r['blas_ms']:.0f}ms  ray={r['shadow_ray_steady_ms']:.2f}ms")
r = render_config("omm_4s_5", omm=True, subdiv=5, fmt=2)
print(f"  4-state: blas={r['blas_ms']:.0f}ms  ray={r['shadow_ray_steady_ms']:.2f}ms")

# === Experiment 5: Stress mode + sample sweep ===
print("\n=== Exp5: Stress mode sample sweep ===")
for n in [4, 16, 32]:
    r = render_config(f"off_s{n}", omm=False, stress=True, samples=n); results.append(r)
    print(f"  OMM-OFF stress s{n}: ray={r['shadow_ray_steady_ms']:.2f}ms")
    r = render_config(f"on_s{n}", omm=True, stress=True, samples=n); results.append(r)
    print(f"  OMM-ON  stress s{n}: ray={r['shadow_ray_steady_ms']:.2f}ms")

# Summary
print("\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)
for r in results:
    print(f"  {r['label']:16s}  blas={r['blas_ms']:6.0f}ms  ray={r['shadow_ray_steady_ms']:.3f}ms  total={r['total_ms']:5.0f}ms  mean={r['img_mean']:.1f}")
