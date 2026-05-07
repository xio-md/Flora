# Dynamic geometry and long-sequence stability

Runnable checks for deformable meshes and instanced particle spheres, plus a long-frame stress run.

## Scripts

- `dynamic_geometry_smoke.py`  
  Short regression: rigid ground + deformable sheet + particles, five geometry updates and renders (`DonutRenderPy`).

- `genesis_dynamic_geometry_smoke.py`  
  Same scenario through direct `GenesisStyleRenderer` (no `DonutRenderPy` wrapper).

- `long_sequence_stability.py`  
  Default **120** frames. Each frame updates the same-topology deformable grid and particle positions/radii, then `update_scene` + `render_frame`.  
  Writes `.temp/long_sequence_stability.json` (repo root relative) with per-frame timings and `rgba_bytes` checks.

## Renderer hardening (`python/rtxns_genesis_style/renderer.py`)

- Deformable: reject non-finite vertex data and out-of-range triangle indices before rebuilding the GLB.
- Particles: reject non-finite centers; clamp radii to a safe range to avoid degenerate or extreme instancing.

## How to read the JSON summary

- `frame_count`, `all_frames_ok`
- `max_update_scene_ms`, `max_render_frame_ms`
- `total_update_scene_ms` / `total_render_frame_ms` for average cost per frame
