"""ReplicaCAD parallel rendering scaling test: N = 1,2,4,8,12 cameras."""
import sys, time, math
from pathlib import Path
_pyd = Path(r"E:\cplus\RTXNS\tools\DonutRenderPyNative.pyd")
if _pyd.exists(): sys.path.insert(0, str(_pyd.parent))
else: sys.path.insert(0, r"E:\cplus\RTXNS\bin\windows-x64")
import DonutRenderPyNative as rr

W, H = 256, 192  # low-res for quick scaling test
STAGE = r"E:\cplus\RTXNS\ReplicaCAD\stages\Stage_v3_sc0_staging.glb"
F = 30
RING_K = 8  # enough for 12 cameras

rr.init(runtime_dir=r"D:\RTXNS", backend="vulkan", device_index=-1, enable_debug=False)
scene = rr.create_scene()
scene.load_scene(STAGE)
scene.set_default_light(direction=[0.3, 0.8, -0.5], color=[1, 1, 1], irradiance=50)
scene.set_ambient(top_rgb=[0.30]*3, bottom_rgb=[0.12]*3)
scene.enable_rt_shadows(False)
scene.set_readback_ring_depth(RING_K)

def add_cams(n):
    ids = []
    for i in range(n):
        a = i * 2 * math.pi / max(n, 1)
        ids.append(scene.add_camera(position=[math.cos(a)*3, 1.5, math.sin(a)*3],
                     target=[0, 1, 0], up=[0, 1, 0],
                     fov_degrees=60, width=W, height=H))
    return ids

# Pre-create all cameras
all_ids = add_cams(12)
print(f"Created {scene.camera_count} cameras, ring depth={scene.readback_ring_depth}")

def bench_sync(ids, n, label):
    scene.render_frame_batch(ids)
    t0 = time.perf_counter()
    for _ in range(F):
        scene.render_frame_batch(ids)
    dt = time.perf_counter() - t0
    batch_ms = 1000 * dt / F
    cam_fps = n * F / dt
    return cam_fps, batch_ms

def bench_async(ids, n, label):
    # Drain ring
    for _ in range(RING_K + 2):
        t = scene.submit_frame_batch(ids)
        if t != 0: scene.read_frame_batch(t)

    in_flight = []
    busy = 0
    t0 = time.perf_counter()
    submitted = 0
    while submitted < F:
        t = scene.submit_frame_batch(ids)
        if t != 0:
            in_flight.append(t)
            submitted += 1
        else:
            busy += 1
            if in_flight:
                scene.read_frame_batch(in_flight.pop(0))

    for t in in_flight:
        scene.read_frame_batch(t)
    dt = time.perf_counter() - t0
    batch_ms = 1000 * dt / F
    cam_fps = n * F / dt
    return cam_fps, batch_ms, busy

print(f"\n{'N':>4}  {'sync_cam-FPS':>14}  {'sync_ms':>9}  {'async_cam-FPS':>15}  {'async_ms':>9}  {'busy':>5}  {'per-cam-FPS':>12}")
print("-" * 85)

for N in [1, 2, 4, 8, 12]:
    ids = all_ids[:N]
    sync_fps, sync_ms = bench_sync(ids, N, f"N={N}")
    async_fps, async_ms, busy = bench_async(ids, N, f"N={N}")
    pc_sync = sync_fps / N
    pc_async = async_fps / N
    print(f"{N:>4}  {sync_fps:>14.1f}  {sync_ms:>8.2f}  {async_fps:>15.1f}  {async_ms:>8.2f}  {busy:>5}  {pc_async:>11.1f}")

# Print speedup summary
print(f"\n{'N':>4}  {'sync_x1':>8}  {'async_x1':>9}  {'async/sync':>11}")
print("-" * 40)
base_sync = None
for N in [1, 2, 4, 8, 12]:
    ids = all_ids[:N]
    sync_fps, _ = bench_sync(ids, N, "")
    async_fps, _, _ = bench_async(ids, N, "")
    if N == 1:
        base_sync = async_fps  # baseline = async cam-fps for N=1
    print(f"{N:>4}  {sync_fps/base_sync:>7.2f}x  {async_fps/base_sync:>8.2f}x  {async_fps/sync_fps:>10.2f}x")

rr.destroy()
