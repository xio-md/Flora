# Week 10: dynamic geometry and long-sequence stability

This folder contains runnable checks for deformable meshes and instanced particle spheres, plus a long-frame stress run.

## Scripts

- `week10_dynamic_geometry_smoke.py`  
  Short regression: rigid ground + deformable sheet + particles, five geometry updates and renders.

- `week10_genesis_dynamic_smoke.py`  
  Same idea through direct `GenesisStyleRenderer` (no `DonutRenderPy` wrapper).

- `week10_long_sequence_stability.py`  
  Default **120** frames (meets the plan’s “100+ frames” bar). Each frame updates the same-topology deformable grid and particle positions/radii, then `update_scene` + `render_frame`.  
  Writes `D:\xmd\RTXNS\.temp\week10_long_sequence_stability.json` with per-frame timings and `rgba_bytes` checks.

## Renderer hardening (`python/rtxns_genesis_style/renderer.py`)

- Deformable: reject non-finite vertex data and out-of-range triangle indices before rebuilding the GLB.
- Particles: reject non-finite centers; clamp radii to a safe range to avoid degenerate or extreme instancing.

## How to report results

Open the JSON summary and quote:

- `frame_count`, `all_frames_ok`
- `max_update_scene_ms`, `max_render_frame_ms`
- `total_update_scene_ms` / `total_render_frame_ms` for average cost per frame
