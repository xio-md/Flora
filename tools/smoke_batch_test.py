"""Week 1 smoke test: old API + new multi-camera API."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(r"D:\RTXNS\bin\windows-x64")))
import DonutRenderPyNative as rr
import numpy as np

SD = Path(r"D:\niagara_bistro")

rr.init(runtime_dir=str(Path(r"D:\RTXNS")), backend="vulkan", device_index=-1, enable_debug=False)
scene = rr.create_scene()
scene.load_scene(str(SD / "bistro.gltf"))

# Test 1: Old API still works
print("[1] Testing old API...")
scene.set_camera(position=[0, 2, 5], target=[0, 1.5, 0], up=[0, 1, 0],
                 fov_degrees=57.3, width=1024, height=768)
scene.set_default_light(direction=[0.3, 0.8, -0.5], color=[1, 1, 1], irradiance=500)
scene.set_ambient(top_rgb=[0.5, 0.5, 0.5], bottom_rgb=[0.2, 0.2, 0.2])
scene.enable_rt_shadows(False)
img = scene.render_frame()
assert len(img) == 1024 * 768 * 4
arr = np.frombuffer(img, np.uint8).reshape(768, 1024, 4)
m = float(arr[:,:,:3].mean())
print(f"  OK: {len(img)} bytes, mean={m:.1f}")

# Test 2: render_frame(0) same as old
print("[2] Testing render_frame(0)...")
img0 = scene.render_frame(0)
assert len(img0) == len(img)
print(f"  OK: {len(img0)} bytes")

# Test 3: camera_count
print(f"[3] camera_count={scene.camera_count}")
assert scene.camera_count == 1

# Test 4: add_camera
print("[4] Testing add_camera...")
cam1 = scene.add_camera(position=[5, 3, 5], target=[0, 1.5, 0], up=[0, 1, 0],
                        fov_degrees=57.3, width=1024, height=768)
print(f"  OK: returned {cam1}, count={scene.camera_count}")
assert cam1 == 1
assert scene.camera_count == 2

# Test 5: render_frame(1)
print("[5] Testing render_frame(1)...")
try:
    img1 = scene.render_frame(1)
    print(f"  OK: {len(img1)} bytes")
except Exception as e:
    print(f"  FAIL: {e}")

# Test 6: render_frame_batch
print("[6] Testing render_frame_batch...")
try:
    imgs = scene.render_frame_batch([0])
    print(f"  OK: batch([0]) -> {len(imgs)} images")
except Exception as e:
    print(f"  FAIL: {e}")

rr.destroy()
print("\nDone.")
