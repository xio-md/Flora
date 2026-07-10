# Optimize Alpha-Tested Shadow for Bistro Foliage

## Current State

Alpha-test infrastructure is fully built but causes GPU timeout on Bistro (2909 instances). The shader, metadata buffers, and geometry flags are all in place. The problem is performance.

## What Was Built

### 1. Combined Geometry Buffers + Metadata (working)
**`D:\RTXNS\src\RayTracedShadow\SceneGeometryProvider.cpp::buildShadowSceneResources`**
- Extracts CPU-side vertex/index/texcoord data before Donut's `FinishedLoading()` frees it
- Builds single combined interleaved VB (pos+uv per vertex) and combined IB
- Builds three StructuredBuffer metadata arrays:
  - `ShadowInstanceMeta` — per-InstanceID: materialIndex, geometryIndex, isAlphaTested
  - `ShadowMaterialMeta` — per-material: alphaCutoff, albedoTextureIndex, domain, flags
  - `ShadowGeometryMeta` — per-geometry: indexByteOffset, vertexByteOffset, texCoordByteOffset

### 2. Alpha-Test Shader (structure ready, disabled due to GPU timeout)
**`D:\RTXNS\src\RayTracedShadow\shaders\shadow_rayquery_cs.hlsl`**
- Uses `RAY_FLAG_NONE` to get all candidate hits (no auto-commit)
- Manual loop over candidates: check `CANDIDATE_NON_OPAQUE_TRIANGLE` type
- For each alpha-tested candidate: reconstruct triangle UV from combined VB/IB, sample alpha texture, compare with alphaCutoff
- Safety: `kMaxIter=64` to prevent infinite loops

### 3. Geometry-Level Opaque Flags (working)
**`D:\RTXNS\src\RayTracedShadow\AccelerationStructure.cpp` line 71-72**
```cpp
if (!geom.isTransparent)
    geoDesc.setFlags(nvrhi::rt::GeometryFlags::Opaque);
```
Opaque triangles auto-commit via BLAS-level `GeometryFlags::Opaque` — only non-opaque triangles reach the manual candidate loop.

### 4. Binding Layout (working)
**`D:\RTXNS\src\RayTracedShadow\RayTracedShadowPass.cpp`**
- Shadow binding layout includes: TLAS, constants, depth, shadow UAV, sampler, 3x StructuredBuffer SRV, 2x RawBuffer SRV, 4x Texture SRV slots
- Dummy buffers/textures created when scene has no alpha-tested geometry

### 5. Shader Bindings
| Register | Resource |
|----------|----------|
| t0 | TLAS |
| b0 | ShadowConstants |
| t1 | Depth texture |
| u0 | Shadow output |
| s0 | Sampler |
| t2 | StructuredBuffer ShadowInstanceMeta |
| t3 | StructuredBuffer ShadowMaterialMeta |
| t4 | StructuredBuffer ShadowGeometryMeta |
| t5 | ByteAddressBuffer combined vertex data |
| t6 | ByteAddressBuffer combined index data |
| t7-t10 | Texture2D alpha textures (kMaxAlphaTextures=4) |

## What Works

- **Genesis box test** (`test_rt_shadow.py`): Opaque-only path with ForceOpaque → localized hard shadows, correct
- **Bistro stable path**: ForceOpaque on all instances, `ACCEPT_FIRST_HIT` → 44% pixels shadowed, no crash
- **Metadata buffers**: Built and uploaded correctly for both Genesis and Bistro
- **Shader compilation**: All HLSL compiles to SPIR-V without errors

## What's Broken

- **Bistro + alpha-test loop**: `RAY_FLAG_NONE` with manual candidate iteration causes GPU timeout (DeviceLost)
- Temporary workaround (currently in code): all instances ForceOpaque, fast opaque-only path

## Files to Modify

| File | Purpose |
|------|---------|
| `D:\RTXNS\src\RayTracedShadow\shaders\shadow_rayquery_cs.hlsl` | Shader — needs performance optimization |
| `D:\RTXNS\src\RayTracedShadow\AccelerationStructure.cpp` | Instance flags — needs selective ForceOpaque |
| `D:\RTXNS\src\RayTracedShadow\SceneGeometryProvider.cpp` | Metadata builder — may need texture index filling |
| `D:\RTXNS\src\RayTracedShadow\RayTracedShadowPass.cpp` | Binding layout — may need dual-pipeline |
| `D:\RTXNS\src\RayTracedShadow\ShadowTypes.h` | Types — constants |

## Optimization Directions

### Option A: Caster/Receiver Instance Mask Separation (Niagara-style)
- Ground/floor instances: mask=0x01 (receiver-only, don't self-shadow)
- Foliage/tree instances: mask=0xFE (caster, tested by shadow ray)
- Shadow ray uses mask=0xFE (skips receiver self-intersection)
- Reduces candidate count by eliminating receiver self-hits
- Re-enable non-ForceOpaque for alpha-tested caster instances only

### Option B: Dual Quality Pipeline (Niagara-style)
- Quality 0 (fast): `ACCEPT_FIRST_HIT` + ForceOpaque for all → opaque-only hard shadows
- Quality 1 (alpha): `RAY_FLAG_NONE` + manual loop for alpha-tested → fine foliage shadows
- Only enable Quality 1 on foliage pixels (via instance mask filtering)
- Reduces the number of rays going through the slow path

### Option C: Hybrid Shader
- Use `RAY_FLAG_NONE` but exit early from the loop after first opaque hit
- Only alpha-test candidates when `isAlphaTested == 1`
- This is what the current shader already does — the overhead is from many candidate iterations per ray

### Option D: Reduce Candidate Overhead
- Limit kMaxIter to something very small (e.g., 8)
- For pixels hitting dense foliage, accept the nearest candidate regardless of alpha
- Acceptable quality loss for performance gain

## Test Commands

```powershell
# Build
$env:Path = "C:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin;$env:Path"
cmake --build D:\RTXNS\build --config Release --target DonutRenderPyNative

# Genesis regression
$env:Path = "${env:LOCALAPPDATA}\Programs\Python\Python312;$env:Path"
$env:PYTHONPATH = "D:\RTXNS\bin\windows-x64"
python D:\RTXNS\tools\test_rt_shadow.py

# Bistro
python D:\RTXNS\tools\test_bistro_shadow.py
```

## Reference

Niagara's alpha-test shadow implementation: `D:\niagara\src\shaders\shadow.comp.glsl` (lines 86-123, `shadowTraceTransparent`)
