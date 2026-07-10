"""Rigorous OMM stress test — 20 frames per config, report mean ± stddev."""
import ctypes, math, struct, sys, numpy as np
from pathlib import Path

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
W, H = 1024, 768
N_FRAMES = 20  # frames to sample per config

def measure(label, omm, stress, samples, n_frames=N_FRAMES):
    """Measure steady-state shadow_ray_ms over N frames, return mean ± stddev."""
    rr.init(runtime_dir=str(repo_root), backend="vulkan", device_index=-1, enable_debug=False)
    s = rr.create_scene()
    s.load_scene(str(sd/"bistro.gltf"))
    s.set_camera(position=pos, target=tgt, up=up, fov_degrees=fov, width=W, height=H, z_near=zn, z_far=200)
    s.set_default_light([-v for v in sun], [1,1,1], 50)
    s.set_ambient([0.03]*3, [0.01]*3)
    s.enable_rt_shadows(True)
    s.enable_omm(omm)
    s.enable_omm_stress(stress)
    s.set_shadow_samples(samples)

    # Frame 0: BLAS build (discard)
    s.render_frame()

    # Frames 1..N: measure
    times = []
    for i in range(n_frames):
        s.render_frame()
        st = s.get_last_frame_stats()
        times.append(st["shadow_ray_ms"])

    rr.destroy()

    arr = np.array(times)
    return arr.mean(), arr.std()

print(f"Measuring over {N_FRAMES} frames per config...\n")
print(f"{'Config':<30s} {'shadow_ray':>10s} {'stddev':>8s}  {'rays/frame'}")
print("-" * 70)

for samples in [4, 16, 32]:
    total_rays = W * H * samples
    m, s = measure(f"OMM-OFF s{samples}", False, True, samples)
    print(f"OMM-OFF stress s{samples:<3d}    {m*1000:6.0f} us  +/-{s*1000:5.0f} us  {total_rays/1e6:.1f}M rays")

for samples in [4, 16, 32]:
    total_rays = W * H * samples
    m, s = measure(f"OMM-ON  s{samples}", True, True, samples)
    print(f"OMM-ON  stress s{samples:<3d}    {m*1000:6.0f} us  +/-{s*1000:5.0f} us  {total_rays/1e6:.1f}M rays")
