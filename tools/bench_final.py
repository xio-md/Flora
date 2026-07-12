import sys,time,math,json;from pathlib import Path
_pyd=Path(r"E:\cplus\RTXNS\tools\DonutRenderPyNative.pyd")
if _pyd.exists():sys.path.insert(0,str(_pyd.parent))
else:sys.path.insert(0,r"E:\cplus\RTXNS\bin\windows-x64")
import DonutRenderPyNative as rr

for W,H,F,label in [(128,96,20,"128x96"),(256,192,20,"256x192"),(512,384,15,"512x384")]:
    RK=8
    print(f"\n=== {label} ===",flush=True)
    rr.init(runtime_dir=r"D:\RTXNS",backend="vulkan",device_index=-1,enable_debug=False)
    s=rr.create_scene()
    s.load_scene(r"E:\cplus\RTXNS\ReplicaCAD\stages\Stage_v3_sc0_staging.glb")
    s.set_default_light(direction=[0.3,0.8,-0.5],color=[1,1,1],irradiance=50)
    s.set_ambient(top_rgb=[0.3]*3,bottom_rgb=[0.12]*3);s.enable_rt_shadows(False);s.set_readback_ring_depth(RK)
    all_ids=[];ptr=0
    for n in [1,2,4,8,12]:
        for i in range(n):a=i*6.283/max(n,1);all_ids.append(s.add_camera(position=[math.cos(a)*3,1.5,math.sin(a)*3],target=[0,1,0],up=[0,1,0],fov_degrees=60,width=W,height=H))
    results=[]
    for N in [1,2,4,8,12]:
        ids=all_ids[ptr:ptr+N];ptr+=N
        s.render_frame_batch(ids)
        t0=time.perf_counter()
        for _ in range(F):s.render_frame_batch(ids)
        dt_sync=time.perf_counter()-t0
        for _ in range(RK+2):t=s.submit_frame_batch(ids);s.read_frame_batch(t) if t!=0 else None
        inflight=[];busy=0;sub=0;t0=time.perf_counter()
        while sub<F:
            t=s.submit_frame_batch(ids)
            if t!=0:inflight.append(t);sub+=1
            else:busy+=1
            if busy>0 and inflight:s.read_frame_batch(inflight.pop(0))
        for t in inflight:s.read_frame_batch(t)
        dt_async=time.perf_counter()-t0
        sfps,afps=N*F/dt_sync,N*F/dt_async
        results.append({"N":N,"sync_fps":round(sfps),"sync_ms":round(1000*dt_sync/F,2),"async_fps":round(afps),"async_ms":round(1000*dt_async/F,2),"per_cam":round(afps/N),"a_s_ratio":round(afps/sfps,2)})
        print(f"  N={N:>2} sync={sfps:>6.0f}fps async={afps:>6.0f}fps per-cam={afps/N:.0f} async/sync={afps/sfps:.2f}x",flush=True)
    rr.destroy()
    print(f"\n{'N':>3} {'sync':>8} {'async':>8} {'ms':>6} {'per-cam':>8} {'a/s':>5}")
    for r in results:print(f"{r['N']:>3} {r['sync_fps']:>8} {r['async_fps']:>8} {r['async_ms']:>6} {r['per_cam']:>8} {r['a_s_ratio']:>5.2f}x")
    Path(r"D:\RTXNS\output\bench_parallel").mkdir(parents=True,exist_ok=True)
    with open(f"D:\\RTXNS\\output\\bench_parallel\\scaling_{label.replace('x','')}.json","w") as f:json.dump(results,f,indent=2)
print("\nDone.")
