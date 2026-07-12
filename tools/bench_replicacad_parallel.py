"""P0 Ring Fix + ReplicaCAD Parallel Rendering Stress Test.

Tests:
  1. Single camera baseline (sync)
  2. N-camera sync batch (Week 2 API)  
  3. N-camera async submit-only throughput (with proper ring drain)
  4. Async end-to-end pipeline (submit + read oldest when busy)
  5. Ring stress: K+2 submits, hash verify EACH frame
  6. Ring depth scaling
"""
import sys, time, hashlib, math
from pathlib import Path
from PIL import Image
import numpy as np

# --- Config ---
STAGE = Path(r"E:\cplus\RTXNS\ReplicaCAD\stages\Stage_v3_sc0_staging.glb")
OUT_DIR = Path(r"D:\RTXNS\output\bench_parallel")
OUT_DIR.mkdir(parents=True, exist_ok=True)
W, H = 512, 384
WARMUP_FRAMES = 5

# Try multiple paths for the pyd
_pyd_dir = Path(__file__).parent
if (_pyd_dir / "DonutRenderPyNative.pyd").exists():
    sys.path.insert(0, str(_pyd_dir))
else:
    sys.path.insert(0, str(Path(r"E:\cplus\RTXNS\bin\windows-x64")))
import DonutRenderPyNative as rr

# --- Helpers ---
def setup_scene():
    rr.init(runtime_dir=str(Path(r"D:\RTXNS")), backend="vulkan", device_index=-1, enable_debug=False)
    scene = rr.create_scene()
    scene.load_scene(str(STAGE))
    scene.set_default_light(direction=[0.3, 0.8, -0.5], color=[1,1,1], irradiance=50)
    scene.set_ambient(top_rgb=[0.30]*3, bottom_rgb=[0.12]*3)
    scene.enable_rt_shadows(False)
    return scene

def add_n_cameras(scene, n, base_dist=3.0):
    indices = []
    for i in range(n):
        angle = i * (2 * math.pi / n)
        px = math.cos(angle) * base_dist
        pz = math.sin(angle) * base_dist
        idx = scene.add_camera(position=[px,1.5,pz], target=[0,1,0], up=[0,1,0],
                               fov_degrees=60, width=W, height=H, z_near=0.1, z_far=50.0)
        indices.append(idx)
    return indices

def img_hash(pixels):
    return hashlib.md5(bytes(pixels)).hexdigest()

def drain_pending(scene, max_tries=20):
    """Read all pending batches to clean the ring."""
    # Use sync render_frame_batch which does submit+wait+read internally
    # and releases ring slots properly
    for _ in range(max_tries):
        try:
            scene.render_frame_batch([0])  # sync: submit 1 camera, wait, read
        except:
            break

# --- Tests ---
def test1_single_baseline(scene):
    print("=" * 60)
    print("Test 1: Single camera sync baseline")
    scene.render_frame(0)  # warmup
    F = 50
    t0 = time.perf_counter()
    for _ in range(F):
        scene.render_frame(0)
    dt = time.perf_counter() - t0
    fps = F / dt
    print(f"  {fps:.1f} FPS ({1000*dt/F:.1f} ms/frame)")
    return fps

def test2_sync_batch(scene, indices, N):
    print(f"\nTest 2: Sync batch render_frame_batch (N={N})")
    scene.render_frame_batch(indices)  # warmup
    F = 30
    t0 = time.perf_counter()
    for _ in range(F):
        scene.render_frame_batch(indices)
    dt = time.perf_counter() - t0
    batch_ms = 1000 * dt / F
    cam_fps = N * F / dt
    print(f"  {cam_fps:.1f} cam-FPS | batch: {batch_ms:.1f} ms | per-cam: {batch_ms/N:.1f} ms")
    return cam_fps, batch_ms

def test3_async_submit_only(scene, indices, N, ring_depth):
    """Async submit-only: measure pure cmdList build+execute cost.
    Properly drains ring when full."""
    print(f"\nTest 3: Async submit-only throughput (N={N}, ringK={ring_depth})")
    
    # Drain ring first
    for _ in range(ring_depth + 2):
        t = scene.submit_frame_batch(indices)
        if t != 0:
            scene.read_frame_batch(t)
    
    F = 30
    in_flight = []
    submit_count = 0
    busy_count = 0
    
    t0 = time.perf_counter()
    while submit_count < F:
        t = scene.submit_frame_batch(indices)
        if t != 0:
            in_flight.append(t)
            submit_count += 1
        else:
            busy_count += 1
            # Drain oldest
            if in_flight:
                scene.read_frame_batch(in_flight.pop(0))
    dt = time.perf_counter() - t0
    
    # Drain remaining
    for t in in_flight:
        scene.read_frame_batch(t)
    
    submit_ms = 1000 * dt / F
    submit_cam_fps = N * F / dt
    print(f"  submit-only: {submit_cam_fps:.0f} cam-FPS | {submit_ms:.1f} ms/batch | busy={busy_count}")
    return submit_cam_fps, busy_count

def test4_async_e2e(scene, indices, N, ring_depth):
    """Async end-to-end: pipeline submit + read, K in-flight max."""
    print(f"\nTest 4: Async end-to-end (N={N}, ringK={ring_depth})")
    
    # Drain ring
    for _ in range(ring_depth + 2):
        t = scene.submit_frame_batch(indices)
        if t != 0:
            scene.read_frame_batch(t)
    
    F = 30
    in_flight = []
    busy_count = 0
    
    t0 = time.perf_counter()
    submitted = 0
    while submitted < F:
        t = scene.submit_frame_batch(indices)
        if t != 0:
            in_flight.append(t)
            submitted += 1
        else:
            busy_count += 1
            if in_flight:
                scene.read_frame_batch(in_flight.pop(0))
    submit_done = time.perf_counter()
    
    # Read all remaining
    for t in in_flight:
        scene.read_frame_batch(t)
    dt = time.perf_counter() - t0
    
    e2e_cam_fps = N * F / dt
    e2e_batch_ms = 1000 * dt / F
    submit_only_ms = 1000 * (submit_done - t0) / F
    read_total_ms = 1000 * (dt - (submit_done - t0))
    
    print(f"  e2e: {e2e_cam_fps:.0f} cam-FPS | batch={e2e_batch_ms:.1f}ms"
          f" | submit_phase={submit_only_ms:.1f}ms read_phase={read_total_ms:.1f}ms"
          f" | busy={busy_count}")
    return e2e_cam_fps, busy_count

def test5_ring_stress(scene, camera_idx, ring_depth):
    """Stress test: submit K+2 frames with unique content, verify no corruption."""
    print(f"\nTest 5: Ring stress K+2 + hash verify (ringK={ring_depth})")
    scene.set_readback_ring_depth(ring_depth)
    K = scene.readback_ring_depth
    assert K == ring_depth, f"Expected K={ring_depth}, got {K}"
    print(f"  Ring depth: {K}")
    
    # Drain ring completely
    drain_pending(scene)
    
    # Use a separate test camera for the stress test to avoid conflict
    # First, compute correct reference hashes using SYNC render (no ring)
    ref_hashes = {}
    base_cam = list([0.0, 1.5, 3.0])
    
    for frame_id in range(K + 2):
        # Unique camera position → unique output
        scene.set_camera_at(camera_idx,
            position=[float(frame_id) * 0.3, 1.5, 3.0],
            target=[0, 1, 0], up=[0,1,0],
            fov_degrees=60, width=W, height=H)
        ref_img = scene.render_frame(camera_idx)  # SYNC: guaranteed correct + frees ring
        ref_hashes[frame_id] = img_hash(ref_img)
    
    # Now async test: submit K+2, check ring occupancy behavior
    tokens = []
    for frame_id in range(K + 2):
        scene.set_camera_at(camera_idx,
            position=[float(frame_id) * 0.3, 1.5, 3.0],
            target=[0, 1, 0], up=[0,1,0],
            fov_degrees=60, width=W, height=H)
        t = scene.submit_frame_batch([camera_idx])
        
        if t != 0:
            tokens.append((frame_id, t))
            print(f"  frame {frame_id}: submitted, token={t}")
        else:
            print(f"  frame {frame_id}: BUSY (expected when >K={K} in-flight)")
    
    # Read back and verify
    errors = 0
    print(f"\n  Verifying {len(tokens)} in-flight frames:")
    for frame_id, token in tokens:
        imgs = scene.read_frame_batch(token)
        actual_hash = img_hash(imgs[0])
        expected = ref_hashes[frame_id]
        status = "OK" if actual_hash == expected else "CORRUPTED!"
        if actual_hash != expected:
            errors += 1
            print(f"  frame {frame_id}: {status} (got {actual_hash[:8]}..., want {expected[:8]}...)")
        else:
            print(f"  frame {frame_id}: {status}  hash={actual_hash[:8]}...")
    
    if errors == 0:
        print(f"\n  *** PASS: All {len(tokens)} frames verified, zero corruption! ***")
    else:
        print(f"\n  *** FAIL: {errors}/{len(tokens)} frames corrupted ***")
    
    # Restore camera
    scene.set_camera_at(camera_idx, position=[0,1.5,3], target=[0,1,0], up=[0,1,0],
                        fov_degrees=60, width=W, height=H)
    return errors == 0, len(tokens)

def test6_ring_depth_scaling(scene, indices, N):
    """Ring depth K=2,4,8 effects on async e2e."""
    print(f"\nTest 6: Ring depth scaling (N={N})")
    results = {}
    for K in [2, 4, 8]:
        scene.set_readback_ring_depth(K)
        drain_pending(scene)
        fps, busy = test4_async_e2e(scene, indices, N, K)
        results[f'K={K}'] = fps
    return results

# --- Main ---
def main():
    print("=" * 60)
    print("RTXNS ReplicaCAD Parallel Rendering Stress Test")
    print(f"Scene: {STAGE.name}  Res: {W}x{H}")
    print("=" * 60)
    
    scene = setup_scene()
    
    # Cameras around scene
    indices_4 = add_n_cameras(scene, 4)
    indices_2 = indices_4[:2]
    indices_1 = [indices_4[0]]
    print(f"\nCreated {len(indices_4)} cameras")
    print(f"Default ring depth: {scene.readback_ring_depth}")
    
    results = {}
    
    # --- Test 1: Baseline ---
    results['single'] = test1_single_baseline(scene)
    
    # --- Test 2: Sync batch ---
    for N, ids in [(2, indices_2), (4, indices_4)]:
        fps, batch_ms = test2_sync_batch(scene, ids, N)
        results[f'sync_N{N}'] = fps
        results[f'sync_N{N}_ms'] = batch_ms
    
    # --- Test 3: Async submit-only ---
    drain_pending(scene)
    for N, ids in [(4, indices_4)]:
        fps, busy = test3_async_submit_only(scene, ids, N, scene.readback_ring_depth)
        results[f'async_submit_N{N}'] = fps
        results[f'async_submit_N{N}_busy'] = busy
    
    # --- Test 4: Async e2e ---
    drain_pending(scene)
    for N, ids in [(4, indices_4)]:
        fps, busy = test4_async_e2e(scene, ids, N, scene.readback_ring_depth)
        results[f'async_e2e_N{N}'] = fps
        results[f'async_e2e_N{N}_busy'] = busy
    
    # --- Test 5: Ring stress ---
    for K, ids in [(2, indices_1), (4, indices_1)]:
        drain_pending(scene)
        ok, n_frames = test5_ring_stress(scene, ids[0], K)
        results[f'ring_K{K}'] = f'{"PASS" if ok else "FAIL"} ({n_frames} ok)'
    
    # --- Test 6: Ring depth scaling ---
    drain_pending(scene)
    scaling = test6_ring_depth_scaling(scene, indices_4, 4)
    results['depth_scaling'] = scaling
    
    # --- Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    rows = [
        ("Single-cam baseline", f"{results['single']:.1f} FPS"),
        ("Sync batch N=2", f"{results['sync_N2']:.1f} cam-FPS ({results['sync_N2_ms']:.1f}ms batch)"),
        ("Sync batch N=4", f"{results['sync_N4']:.1f} cam-FPS ({results['sync_N4_ms']:.1f}ms batch)"),
        ("Async submit-only N=4", f"{results['async_submit_N4']:.0f} cam-FPS (busy={results['async_submit_N4_busy']})"),
        ("Async e2e N=4", f"{results['async_e2e_N4']:.0f} cam-FPS (busy={results['async_e2e_N4_busy']})"),
        ("Ring stress K=2", results['ring_K2']),
        ("Ring stress K=4", results['ring_K4']),
        ("Depth scaling", str(results['depth_scaling'])),
    ]
    for label, val in rows:
        print(f"  {label:<30} {val}")
    
    # Speedup ratios
    print(f"\n  Speedup ratios vs single-cam:")
    print(f"    sync N=4 / single:  {results['sync_N4'] / results['single']:.2f}x")
    print(f"    async e2e N=4 / single: {results['async_e2e_N4'] / results['single']:.2f}x")
    
    # Save reference image
    ref_bytes = scene.render_frame(indices_4[0])
    ref_img = np.frombuffer(ref_bytes, np.uint8).reshape(H, W, 4)[:,:,:3]
    Image.fromarray(ref_img, "RGB").save(OUT_DIR / "replicacad_cam0.png")
    print(f"\nReference image: {OUT_DIR / 'replicacad_cam0.png'}")
    
    rr.destroy()
    print("Done.")

if __name__ == "__main__":
    main()
