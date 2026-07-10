"""OMM steady-state performance test."""
import ctypes, math, struct, sys, time
from pathlib import Path
repo_root = Path(r"D:\RTXNS")
sys.path.insert(0, str(repo_root / "bin" / "windows-x64"))
import DonutRenderPyNative as rr

def _rotate_by_quat_xyzw(q, v):
    x, y, z, w = q
    cx, cy, cz = y*v[2]-z*v[1], z*v[0]-x*v[2], x*v[1]-y*v[0]
    tx, ty, tz = 2*cx, 2*cy, 2*cz
    return [v[0]+w*tx+y*tz-z*ty, v[1]+w*ty+z*tx-x*tz, v[2]+w*tz+x*ty-y*tx]

class _NC(ctypes.Structure):
    _fields_=[("pos",ctypes.c_float*3),("ori",ctypes.c_float*4),("fov_y",ctypes.c_float),("znear",ctypes.c_float)]
class _NH(ctypes.Structure):
    _fields_=[("magic",ctypes.c_uint32),("ver",ctypes.c_uint32),("mmv",ctypes.c_uint32),("mmt",ctypes.c_uint32),
              ("clrt",ctypes.c_bool),("comp",ctypes.c_bool),("cvb",ctypes.c_uint32),("cib",ctypes.c_uint32),
              ("cmd",ctypes.c_uint32),("cmv",ctypes.c_uint32),("vc",ctypes.c_uint32),("ic",ctypes.c_uint32),
              ("mc",ctypes.c_uint32),("mdc",ctypes.c_uint32),("mvc",ctypes.c_uint32),("meshc",ctypes.c_uint32),
              ("matc",ctypes.c_uint32),("dc",ctypes.c_uint32),("tpc",ctypes.c_uint32),
              ("oads",ctypes.c_uint32),("oids",ctypes.c_uint32),("odc",ctypes.c_uint32),("oms",ctypes.c_uint32),
              ("cam",_NC),("sd",ctypes.c_float*3)]

def load_niagara(scene_dir):
    sd=Path(scene_dir); hb=(sd/"bistro.gltf.cache").read_bytes()[:ctypes.sizeof(_NH)]
    h=_NH.from_buffer_copy(hb)
    if h.magic!=0x434E4353 or h.ver!=5: raise RuntimeError("bad cache")
    cb=(sd/"bistro.gltf.camera").read_bytes()
    cv,px,py,pz,qx,qy,qz,qw=struct.unpack("<I3f4f",cb)
    p=[px,py,pz]; f=_rotate_by_quat_xyzw([qx,qy,qz,qw],[0,0,-1])
    u=_rotate_by_quat_xyzw([qx,qy,qz,qw],[0,1,0])
    t=[p[i]+f[i] for i in range(3)]
    return p,t,u,float(h.cam.fov_y)*180/math.pi,float(h.cam.znear),[float(v) for v in h.sd]

sd = Path(r"D:\niagara_bistro")
pos, tgt, up, fov, zn, sun = load_niagara(sd)

def render_scene(label, enable_omm):
    rr.init(runtime_dir=str(repo_root), backend="vulkan", device_index=-1, enable_debug=False)
    s = rr.create_scene()
    s.load_scene(str(sd / "bistro.gltf"))
    s.set_camera(position=pos, target=tgt, up=up, fov_degrees=fov,
                 width=1024, height=768, z_near=zn, z_far=200.0)
    s.set_default_light(direction=[-v for v in sun], color=[1,1,1], irradiance=50.0)
    s.set_ambient(top_rgb=[0.03,0.03,0.03], bottom_rgb=[0.01,0.01,0.01])
    s.enable_rt_shadows(True)
    s.enable_omm(enable_omm)

    # Frame 0: first frame (BLAS build + OMM baking)
    t0 = time.time()
    s.render_frame()
    s0 = s.get_last_frame_stats()
    print(f"  {label} frame0: total={s0['total_ms']:.0f}ms blas={s0['blas_build_ms']:.0f}ms shadow_ray={s0['shadow_ray_ms']:.1f}ms")

    # Frame 1-3: steady state
    for i in range(1, 4):
        s.render_frame()
        st = s.get_last_frame_stats()
        print(f"  {label} frame{i}: total={st['total_ms']:.1f}ms shadow_ray={st['shadow_ray_ms']:.1f}ms")

    rr.destroy()

print("=== OMM OFF ===")
render_scene("OMM-OFF", False)
print("\n=== OMM ON ===")
render_scene("OMM-ON", True)
