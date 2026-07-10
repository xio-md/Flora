"""OMM experiment — with disk cache to skip re-baking.

Usage:
  python test_omm_simple.py              # auto: bake if no cache, else load
  python test_omm_simple.py --bake       # force re-bake and save cache
  python test_omm_simple.py --no-omm     # skip OMM entirely (quick baseline)
"""
import ctypes, math, struct, sys, time, numpy as np
from pathlib import Path
from PIL import Image

repo_root = Path(r"D:\RTXNS")
sys.path.insert(0, str(repo_root / "bin" / "windows-x64"))
import DonutRenderPyNative as rr

# --- Parse args ---
force_bake = "--bake" in sys.argv
no_omm = "--no-omm" in sys.argv

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
omm_cache = out / "omm_cache.bin"
subdiv, fmt = 2, 2  # subdiv=2 (16 microtris), 4-state format

# === Phase 1: OMM OFF baseline ===
rr.init(runtime_dir=str(repo_root), backend="vulkan", device_index=-1, enable_debug=False)
scene = rr.create_scene()
scene.load_scene(str(sd/"bistro.gltf"))
scene.set_camera(position=pos, target=tgt, up=up, fov_degrees=fov, width=W, height=H, z_near=zn, z_far=200.0)
scene.set_default_light([-v for v in sun], [1,1,1], 50)
scene.set_ambient([0.03]*3, [0.01]*3)

# No Shadow
scene.enable_rt_shadows(False)
img_ns = scene.render_frame()
arr_ns = np.frombuffer(img_ns, np.uint8).reshape(H,W,4)[:,:,:3]
Image.fromarray(arr_ns, "RGB").save(out/"exp_noshadow.png")
print(f"NoShadow: mean={arr_ns.mean():.1f}")

# OMM OFF
scene.enable_rt_shadows(True)
img_off = scene.render_frame()
st = scene.get_last_frame_stats()
arr_off = np.frombuffer(img_off, np.uint8).reshape(H,W,4)[:,:,:3]
Image.fromarray(arr_off, "RGB").save(out/"exp_omm_off2.png")
print(f"OMM OFF: mean={arr_off.mean():.1f} total={st['total_ms']:.0f}ms blas={st['blas_build_ms']:.0f}ms ray={st['shadow_ray_ms']:.2f}ms")

# Steady state OMM OFF
img_off2 = scene.render_frame()
st2 = scene.get_last_frame_stats()
print(f"OMM OFF steady: total={st2['total_ms']:.1f}ms ray={st2['shadow_ray_ms']:.2f}ms")

rr.destroy()

if no_omm:
    print("\n--no-omm: skipping OMM ON phase")
    sys.exit(0)

# === Phase 2: OMM ON (with cache) ===
rr.init(runtime_dir=str(repo_root), backend="vulkan", device_index=-1, enable_debug=False)
scene2 = rr.create_scene()
scene2.load_scene(str(sd/"bistro.gltf"))
scene2.set_camera(position=pos, target=tgt, up=up, fov_degrees=fov, width=W, height=H, z_near=zn, z_far=200.0)
scene2.set_default_light([-v for v in sun], [1,1,1], 50)
scene2.set_ambient([0.03]*3, [0.01]*3)
scene2.enable_rt_shadows(True)
scene2.enable_omm(True)
scene2.set_omm_config(subdiv, fmt)

# Try loading cache (skip CPU bake if available)
cache_loaded = False
if not force_bake and omm_cache.exists():
    cache_loaded = scene2.load_omm_cache(str(omm_cache))

print(f"Rendering OMM ON ({'from cache' if cache_loaded else 'baking'})...")
t0 = time.time()
img_on = scene2.render_frame()
t1 = time.time()
st_on = scene2.get_last_frame_stats()
arr_on = np.frombuffer(img_on, np.uint8).reshape(H,W,4)[:,:,:3]
Image.fromarray(arr_on, "RGB").save(out/"exp_omm_on2.png")
print(f"OMM ON:  mean={arr_on.mean():.1f} total={st_on['total_ms']:.0f}ms blas={st_on['blas_build_ms']:.0f}ms ray={st_on['shadow_ray_ms']:.2f}ms prep={t1-t0:.1f}s")

# Save cache if we just baked
if not cache_loaded:
    scene2.save_omm_cache(str(omm_cache))
    print(f"OMM cache saved to {omm_cache}")

# Steady state OMM ON
img_on2 = scene2.render_frame()
st_on2 = scene2.get_last_frame_stats()
arr_on2 = np.frombuffer(img_on2, np.uint8).reshape(H,W,4)[:,:,:3]
Image.fromarray(arr_on2, "RGB").save(out/"exp_omm_on_steady.png")
print(f"OMM ON steady: mean={arr_on2.mean():.1f} total={st_on2['total_ms']:.1f}ms ray={st_on2['shadow_ray_ms']:.2f}ms")

# Diff
diff = np.abs(arr_on.astype(float) - arr_off.astype(float)).max(axis=2)
diff_vis = np.clip(diff * 20, 0, 255).astype(np.uint8)
Image.fromarray(diff_vis, "L").save(out/"exp_omm_diff2.png")
print(f"Pixel diff: max={diff.max():.0f} mean={diff.mean():.2f}")

rr.destroy()
