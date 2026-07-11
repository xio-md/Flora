"""Week 3 smoke test: async batch API."""
import sys, time, math
from pathlib import Path
sys.path.insert(0, str(Path(r"D:\RTXNS\bin\windows-x64")))
import DonutRenderPyNative as rr

SD = Path(r"D:\niagara_bistro")
W, H = 1024, 768

rr.init(runtime_dir=str(Path(r"D:\RTXNS")), backend="vulkan", device_index=-1, enable_debug=False)
scene = rr.create_scene()
scene.load_scene(str(SD / "bistro.gltf"))
scene.set_default_light(direction=[0.3,0.8,-0.5], color=[1,1,1], irradiance=500)
scene.set_ambient(top_rgb=[0.5]*3, bottom_rgb=[0.2]*3)
scene.enable_rt_shadows(False)

# Set camera 0
scene.set_camera(position=[0,2,5], target=[0,1.5,0], up=[0,1,0], fov_degrees=57.3, width=W, height=H)
# Add camera 1, 2
scene.add_camera(position=[5,2,0], target=[0,1.5,0], up=[0,1,0], fov_degrees=57.3, width=W, height=H)
scene.add_camera(position=[0,2,-5], target=[0,1.5,0], up=[0,1,0], fov_degrees=57.3, width=W, height=H)
print(f"Created {scene.camera_count} cameras")

# Test 1: sync batch (backward compat)
print("[1] Sync batch...")
imgs = scene.render_frame_batch([0, 1, 2])
print(f"    OK: {len(imgs)} images, each {len(imgs[0])} bytes")

# Test 2: async submit + read
print("[2] Async submit + read...")
token = scene.submit_frame_batch([0, 1, 2])
print(f"    Token: {token}")
ready = scene.is_batch_ready(token)
print(f"    is_batch_ready: {ready}")
imgs2 = scene.read_frame_batch(token)
print(f"    OK: {len(imgs2)} images, each {len(imgs2[0])} bytes")

# Test 3: submit multiple batches before reading
print("[3] Pipelined submit -> submit -> read -> read...")
t1 = scene.submit_frame_batch([0])
t2 = scene.submit_frame_batch([1])
r1 = scene.read_frame_batch(t1)
r2 = scene.read_frame_batch(t2)
print(f"    OK: batch1={len(r1)}, batch2={len(r2)}")

# Test 4: benchmark async vs sync
print("[4] Benchmark async vs sync...")
F = 20
indices = [0, 1, 2]

# Sync baseline
scene.render_frame_batch(indices)
t0 = time.perf_counter()
for _ in range(F):
    scene.render_frame_batch(indices)
t1 = time.perf_counter()
sync_fps = 3*F/(t1-t0)
print(f"    Sync:  {sync_fps:.0f} FPS ({1000*(t1-t0)/F:.1f}ms/batch)")

# Async: submit all, then read all
tokens = []
t0 = time.perf_counter()
for _ in range(F):
    tokens.append(scene.submit_frame_batch(indices))
t_submit = time.perf_counter()
for t in tokens:
    scene.read_frame_batch(t)
t1 = time.perf_counter()
async_fps = 3*F/(t1-t0)
print(f"    Async: {async_fps:.0f} FPS (submit {1000*(t_submit-t0)/F:.1f}ms, total {1000*(t1-t0)/F:.1f}ms/batch)")

rr.destroy()
print("\nALL TESTS PASSED")
