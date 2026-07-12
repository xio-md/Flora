"""Quick ring correctness verification test."""
import sys, hashlib
from pathlib import Path
_pyd = Path(r"E:\cplus\RTXNS\tools\DonutRenderPyNative.pyd")
if _pyd.exists():
    sys.path.insert(0, str(_pyd.parent))
else:
    sys.path.insert(0, r"E:\cplus\RTXNS\bin\windows-x64")
import DonutRenderPyNative as rr

W, H = 256, 192
STAGE = r"E:\cplus\RTXNS\ReplicaCAD\stages\Stage_v3_sc0_staging.glb"

rr.init(runtime_dir=r"D:\RTXNS", backend="vulkan", device_index=-1, enable_debug=False)
scene = rr.create_scene()
scene.load_scene(STAGE)
scene.set_default_light(direction=[0.3, 0.8, -0.5], color=[1, 1, 1], irradiance=50)
scene.set_ambient(top_rgb=[0.3]*3, bottom_rgb=[0.12]*3)
scene.enable_rt_shadows(False)
scene.set_readback_ring_depth(4)

cam0 = scene.add_camera(position=[0, 1.5, 3], target=[0, 1, 0], up=[0, 1, 0],
                        fov_degrees=60, width=W, height=H)
print(f"Ring depth: {scene.readback_ring_depth}")

# --- Drain ring ---
for _ in range(6):
    t = scene.submit_frame_batch([cam0])
    if t != 0:
        scene.read_frame_batch(t)
print("Ring drained.")

# --- Build reference hashes (sync render = ground truth) ---
ref_hashes = {}
for fid in range(6):  # K+2 = 6
    scene.set_camera_at(cam0, position=[fid * 0.2, 1.5, 3.0],
                        target=[0, 1, 0], up=[0, 1, 0],
                        fov_degrees=60, width=W, height=H)
    img = scene.render_frame(cam0)  # sync = correct
    ref_hashes[fid] = hashlib.md5(bytes(img)).hexdigest()
    print(f"  ref frame {fid}: hash={ref_hashes[fid][:12]}...")

# --- Async stress: K+2 submits ---
tokens = []
for fid in range(6):
    scene.set_camera_at(cam0, position=[fid * 0.2, 1.5, 3.0],
                        target=[0, 1, 0], up=[0, 1, 0],
                        fov_degrees=60, width=W, height=H)
    t = scene.submit_frame_batch([cam0])
    if t != 0:
        tokens.append((fid, t))
        print(f"  submit frame {fid}: token={t}")
    else:
        print(f"  submit frame {fid}: BUSY (ring at capacity K=4)")

# --- Verify all submitted frames ---
errors = 0
for fid, token in tokens:
    imgs = scene.read_frame_batch(token)
    actual = hashlib.md5(bytes(imgs[0])).hexdigest()
    expected = ref_hashes[fid]
    ok = actual == expected
    if not ok:
        errors += 1
    status = "OK" if ok else "CORRUPTED"
    print(f"  verify {fid}: got={actual[:12]}... exp={expected[:12]}... {status}")

if errors == 0:
    print(f"\n*** Ring Stress PASS: {len(tokens)} frames, zero corruption! ***")
else:
    print(f"\n*** Ring Stress FAIL: {errors} corrupted frames! ***")

rr.destroy()
