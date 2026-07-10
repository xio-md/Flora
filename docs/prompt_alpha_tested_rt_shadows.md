# Implement Alpha-Tested Ray-Traced Shadows for Bistro Foliage

## Goal

Fix RTXNS Bistro ray-traced shadow quality so tree/foliage shadows retain the fine alpha-mask detail visible in Niagara's reference image:

- RTXNS current output: `D:\RTXNS\output\bistro_test\bistro_rt_shadow.png`
- Niagara reference: `D:\niagara_bistro\gt.png`

This is **not** primarily a shadow blur or sun jitter problem. Niagara can render the reference with sun jitter/blur disabled. The main missing feature is alpha-tested foliage shadowing.

## Root Cause

RTXNS currently ray-traces shadows against triangle geometry only. Alpha-tested leaf cards are treated as solid triangles, so many leaf shadows merge into large dark blobs.

Niagara does not do that. In `D:\niagara\src\shaders\shadow.comp.glsl`, the high-quality path uses `shadowTraceTransparent`:

1. Ray query finds a candidate triangle.
2. It reads `instanceId`, `primitiveIndex`, and barycentrics.
3. It loads the draw/material/mesh data.
4. It reconstructs the triangle UVs.
5. It samples the material albedo alpha.
6. It confirms the hit only if `alpha >= 0.5`; otherwise the ray continues.

Relevant Niagara code:

```glsl
while (rayQueryProceedEXT(rq))
{
    int objid = rayQueryGetIntersectionInstanceIdEXT(rq, false);
    int triid = rayQueryGetIntersectionPrimitiveIndexEXT(rq, false);
    vec2 bary = rayQueryGetIntersectionBarycentricsEXT(rq, false);

    MeshDraw draw = draws[objid];
    Material material = materials[draw.materialIndex];
    Mesh mesh = meshes[draw.meshIndex];

    uint vertexOffset = mesh.vertexOffset;
    uint indexOffset = mesh.lods[mesh.lodRT].indexOffset;

    uint tria = indices[indexOffset + triid * 3 + 0];
    uint trib = indices[indexOffset + triid * 3 + 1];
    uint tric = indices[indexOffset + triid * 3 + 2];

    vec2 uva = vec2(vertices[vertexOffset + tria].tu, vertices[vertexOffset + tria].tv);
    vec2 uvb = vec2(vertices[vertexOffset + trib].tu, vertices[vertexOffset + trib].tv);
    vec2 uvc = vec2(vertices[vertexOffset + tric].tu, vertices[vertexOffset + tric].tv);

    vec2 uv = uva * (1 - bary.x - bary.y) + uvb * bary.x + uvc * bary.y;

    float alpha = 1.0;
    if (material.albedoTexture > 0)
        alpha = textureLod(SAMP(material.albedoTexture), uv, 0).a;

    if (alpha >= 0.5)
        rayQueryConfirmIntersectionEXT(rq);
}
```

RTXNS needs the same semantic behavior.

## Important Constraints

- Do **not** solve this by adding blur.
- Do **not** solve this by enabling `sunJitter`.
- Do **not** simply skip transparent geometry. That removes the leaf shadows entirely and is not acceptable.
- Keep `sunJitter = 0.0f` for the Bistro reference path.
- Preserve opaque geometry performance where reasonable.
- The final image should have detailed foliage shadows, not giant solid blobs and not missing leaves.

## Files To Inspect First

RTXNS:

- `D:\RTXNS\src\RayTracedShadow\shaders\shadow_rayquery_cs.hlsl`
- `D:\RTXNS\src\RayTracedShadow\RayTracedShadowPass.h`
- `D:\RTXNS\src\RayTracedShadow\RayTracedShadowPass.cpp`
- `D:\RTXNS\src\RayTracedShadow\AccelerationStructure.cpp`
- `D:\RTXNS\src\RayTracedShadow\SceneGeometryProvider.cpp`
- `D:\RTXNS\src\RayTracedShadow\ShadowTypes.h`
- `D:\RTXNS\src\PythonBindings\headless_pbr.cpp`

Useful Donut references:

- `D:\RTXNS\external\donut\include\donut\engine\Scene.h`
- `D:\RTXNS\external\donut\include\donut\engine\SceneTypes.h`
- `D:\RTXNS\external\donut\include\donut\shaders\bindless.h`
- `D:\RTXNS\external\donut\include\donut\shaders\material_cb.h`
- `D:\RTXNS\external\donut\include\donut\shaders\scene_material.hlsli`
- `D:\RTXNS\external\donut\src\engine\Scene.cpp`
- `D:\RTXNS\external\donut\src\engine\Material.cpp`
- `D:\RTXNS\external\donut\src\engine\TextureCache.cpp`
- `D:\RTXNS\external\donut\include\donut\engine\DescriptorTableManager.h`

Niagara reference:

- `D:\niagara\src\shaders\shadow.comp.glsl`
- `D:\niagara\src\scenert.cpp`
- `D:\niagara\src\scene.cpp`
- `D:\niagara\src\scene.h`

## Current State And Known Pitfall

RTXNS already detects transparent-ish material domains in `SceneGeometryProvider.cpp`:

```cpp
desc.isTransparent =
    geom->material->domain == donut::engine::MaterialDomain::AlphaBlended ||
    geom->material->domain == donut::engine::MaterialDomain::AlphaTested ||
    geom->material->domain == donut::engine::MaterialDomain::Transmissive;
```

But this is not enough. The current ray query shader does not know material alpha or UVs.

Also watch for this bug pattern in `AccelerationStructure.cpp`:

```cpp
instance.setFlags(nvrhi::rt::InstanceFlags::ForceOpaque);
```

If this is applied to alpha-tested foliage instances, candidate intersections cannot be alpha-tested correctly and the leaf card becomes solid.

However, merely removing `ForceOpaque` or skipping transparent geometry is also not enough. The correct fix is to implement candidate-hit alpha testing.

## Recommended Implementation Plan

### Step 1: Revert Any Diagnostic "Skip Transparent Geometry" Fix

If current code avoids `ForceOpaque` for transparent geometry but does not implement alpha testing, it will remove too much foliage shadow. That may be useful as a diagnostic but should not be the final behavior.

Final behavior should be:

- Opaque geometry: can be marked opaque and can use fast terminate-on-first-hit behavior.
- Alpha-tested geometry: must be non-opaque enough for ray query candidate intersections, then shader must alpha-test candidate hits.

### Step 2: Enable/Expose Donut Bindless Scene Resources If Possible

The cleanest route is to reuse Donut's existing bindless scene data:

- `Scene::GetMaterialBuffer()`
- `Scene::GetGeometryBuffer()`
- `Scene::GetInstanceBuffer()`
- `Scene::GetDescriptorTable()`

Donut has shader helpers in `donut/shaders/bindless.h`:

- `LoadGeometryData`
- `LoadInstanceData`
- `LoadMaterialConstants`
- `MaterialConstants.baseOrDiffuseTextureIndex`
- `MaterialConstants.opacityTextureIndex`
- `MaterialConstants.alphaCutoff`
- `MaterialConstants.flags`

But the current Python renderer appears to create `TextureCache` and `Scene` with `nullptr` descriptor table in `headless_pbr.cpp`, which disables bindless scene resources:

```cpp
m_texture_cache = std::make_shared<TextureCache>(m_context->device(), m_native_fs, nullptr);
...
m_scene = std::make_unique<Scene>(
    device,
    *m_context->shader_factory(),
    m_native_fs,
    m_texture_cache,
    nullptr,
    nullptr);
```

Please investigate how to create an NVRHI bindless layout and `DescriptorTableManager` for Vulkan. Likely pieces:

- `nvrhi::BindlessLayoutDesc`
- `device->createBindlessLayout(...)`
- `donut::engine::DescriptorTableManager`
- pass that descriptor table manager into both `TextureCache` and `Scene`

The goal is that loaded textures have valid `bindlessDescriptor`, and `Scene` creates material/geometry/instance buffers.

### Step 3: Pass Scene Material/Geometry Resources To RayTracedShadowPass

Extend `RayTracedShadowPass::renderShadow(...)` or add a small `ShadowSceneResources` struct so the pass can bind:

- TLAS
- shadow constants buffer
- depth texture
- shadow output UAV
- material buffer
- geometry buffer
- instance buffer
- any raw vertex/index buffers needed for UV reconstruction
- bindless descriptor table or texture array/sampler for alpha sampling

Prefer using Donut's bindless buffers if available.

If Donut bindless resource binding is too hard, a fallback is acceptable:

- Build a small RTXNS-specific metadata buffer mapping ray query `InstanceID` and `PrimitiveIndex` to:
  - index buffer
  - vertex buffer
  - texcoord offset
  - material alpha cutoff
  - base/diffuse alpha texture descriptor index
  - opacity texture descriptor index if used

But do not hardcode Bistro-specific asset names.

### Step 4: Modify `shadow_rayquery_cs.hlsl` To Alpha-Test Candidate Hits

Replace the current one-shot logic:

```hlsl
RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> query;
query.TraceRayInline(...);
query.Proceed();
float shadow = (query.CommittedStatus() == COMMITTED_TRIANGLE_HIT) ? 0.0f : 1.0f;
```

with logic equivalent to Niagara:

1. Trace ray without forcing all geometry opaque.
2. Loop over candidate intersections.
3. For opaque candidate hits, accept.
4. For alpha-tested/transmissive/alpha-blended candidate hits:
   - get candidate instance ID
   - get candidate primitive index
   - get candidate barycentrics
   - reconstruct triangle UV
   - sample opacity/base alpha texture
   - compare to material alpha cutoff, default `0.5`
   - confirm hit only if alpha passes
5. If no hit is confirmed, pixel is lit.

HLSL ray query APIs to investigate/use:

- `CandidateType()`
- `CandidateInstanceID()` or equivalent for inline ray query
- `CandidatePrimitiveIndex()`
- `CandidateTriangleBarycentrics()`
- `CommitNonOpaqueTriangleHit()` or equivalent
- `CommittedStatus()`

Use the exact API names supported by DXC for HLSL ray query.

### Step 5: Correct AS Flags

In `AccelerationStructure.cpp`, ensure:

- Opaque geometry can use `nvrhi::rt::GeometryFlags::Opaque`.
- Alpha-tested/non-opaque geometry must not be forced opaque in a way that prevents candidate alpha testing.
- Do not set `InstanceFlags::ForceOpaque` on instances containing alpha-tested foliage if the shader needs candidate intersections.

Be careful: if one BLAS/instance contains both opaque and alpha-tested sub-geometries, instance-level `ForceOpaque` may be too coarse. Geometry-level flags are preferable.

### Step 6: Keep No-Jitter Reference Path

In `headless_pbr.cpp`, keep:

```cpp
shadowConstants.sunJitter = 0.0f;
```

Do not call a blur pass for this task.

## Build And Test

Build:

```powershell
& 'C:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' --build D:\RTXNS\build --config Release --target DonutRenderPyNative
```

Run Bistro:

```powershell
$env:PYTHONPATH = 'D:\RTXNS\bin\windows-x64'
python D:\RTXNS\tools\test_bistro_shadow.py
```

Run regression:

```powershell
$env:PYTHONPATH = 'D:\RTXNS\bin\windows-x64'
python D:\RTXNS\tools\test_rt_shadow.py
```

The Bistro test alone does not judge visual correctness. After it runs, inspect:

```text
D:\RTXNS\output\bistro_test\bistro_rt_shadow.png
```

Compare against:

```text
D:\niagara_bistro\gt.png
```

Expected qualitative result:

- Foliage shadows should have many fine holes/details.
- They should not be solid dark blobs.
- They should not disappear almost entirely.
- No blur/jitter should be needed to see the alpha-mask detail.

## Acceptance Criteria

1. Code builds.
2. `test_bistro_shadow.py` runs without errors.
3. `test_rt_shadow.py` runs without errors.
4. Bistro tree/canopy shadow detail is much closer to `D:\niagara_bistro\gt.png`.
5. `sunJitter` remains `0.0f` for this comparison.
6. No shadow blur is introduced as the primary fix.
7. Transparent foliage is not skipped; it casts alpha-masked shadows.

## Notes For The Implementer

- This task is about correctness first, performance second.
- It is acceptable if the first working version uses the high-quality alpha-test path for all shadow rays, as long as the image is correct.
- Once correct, we can optimize by using opaque fast path for fully opaque geometry.
- If bindless setup is too large, implement the smallest generic metadata/resource path needed for Bistro alpha-tested foliage, but keep it general enough for other alpha-tested glTF scenes.
- Please document any assumptions about Donut material texture indices, descriptor tables, and UV buffer layout.
