"""Stable multi-cmdList A/B benchmark on one Graphics queue."""
import sys,time,math,statistics;from pathlib import Path
_pyd=Path(r"E:\cplus\RTXNS\tools\DonutRenderPyNative.pyd")
if _pyd.exists():sys.path.insert(0,str(_pyd.parent))
else:sys.path.insert(0,r"E:\cplus\RTXNS\bin\windows-x64")
import DonutRenderPyNative as rr

W,H,F,RK,WARMUP,TRIALS=256,192,120,8,24,5
rr.init(runtime_dir=r"D:\RTXNS",backend="vulkan",device_index=-1,enable_debug=False)
s=rr.create_scene();s.load_scene(r"E:\cplus\RTXNS\ReplicaCAD\stages\Stage_v3_sc0_staging.glb")
s.set_default_light(direction=[0.3,0.8,-0.5],color=[1,1,1],irradiance=50)
s.set_ambient(top_rgb=[0.3]*3,bottom_rgb=[0.12]*3);s.enable_rt_shadows(False);s.set_readback_ring_depth(RK)

ids12=[]
for i in range(12):a=i*6.283/12;ids12.append(s.add_camera(position=[math.cos(a)*3,1.5,math.sin(a)*3],target=[0,1,0],up=[0,1,0],fov_degrees=60,width=W,height=H))

def submit(ids, mb):
    return s.submit_frame_batch_ex(ids,mb)

def bench_once(ids, mb):
    # Warm up without leaving pending GPU work behind.
    for _ in range(WARMUP):
        token = submit(ids, mb)
        if token == 0: raise RuntimeError("Ring unexpectedly busy during warmup")
        s.read_frame_batch(token)

    # Keep exactly up to RK batches in flight. This avoids measuring the busy
    # return path and makes every mode use the same submit/readback schedule.
    inflight=[]
    t0=time.perf_counter()
    for _ in range(F):
        if len(inflight) == RK:
            s.read_frame_batch(inflight.pop(0))
        token = submit(ids, mb)
        if token == 0: raise RuntimeError("Ring unexpectedly busy during measurement")
        inflight.append(token)
    for token in inflight:s.read_frame_batch(token)
    dt=time.perf_counter()-t0
    return len(ids)*F/dt, 1000*dt/F

def bench_suite(ids, specs):
    samples={label:[] for label,_ in specs}
    for trial in range(TRIALS):
        order=specs if trial%2==0 else list(reversed(specs))
        for label,mb in order:
            fps,ms=bench_once(ids,mb)
            samples[label].append((fps,ms))
    medians={label:(statistics.median(v[0] for v in values),statistics.median(v[1] for v in values))
             for label,values in samples.items()}
    return medians

specs=[("single cmdList (mb=0)",0),("mb=4",4),("mb=2",2)]
print(f"\nMulti-cmdList A/B, {W}x{H}, ringK={RK}, {TRIALS} interleaved trials x {F} batches\n",flush=True)

results12=bench_suite(ids12,specs)
base=results12["single cmdList (mb=0)"][0]
print("N=12 median end-to-end results:")
for label,_ in specs:
    fps,ms=results12[label]
    print(f"  {label:<24} {fps:>7.0f} cam-FPS  {ms:.2f}ms/batch  vs baseline={fps/base-1:+.1%}")

print("\nN=8 median end-to-end results:",flush=True)
ids8=ids12[:8]
results8=bench_suite(ids8,specs)
base=results8["single cmdList (mb=0)"][0]
for label,_ in specs:
    fps,ms=results8[label]
    print(f"  {label:<24} {fps:>7.0f} cam-FPS  {ms:.2f}ms/batch  vs baseline={fps/base-1:+.1%}")

rr.destroy()
